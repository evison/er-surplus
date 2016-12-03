//[[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <iostream>
#include <string>
#include <fstream>
using namespace arma;
using namespace Rcpp;
using namespace std;


/**
* consumer surplus maximization through KPR Utility model
*/
class CSMax {
public:
	List inputData;
	uvec users, items, quantities;
	vec prices;

	int numDim;
	int numUsers;
	int numItems;
	int numRows;
	/// user and item frequency
	uvec userFreq, itemFreq;

	mat uMat, iMat;
	vec uBias, iBias;
	double globalBias;

	/// hyper parameters
	double lambda, biasLambda;

	List summary;
public:
	CSMax(int dim = 5, double lambda = 0.01, double biasLambda = 0.01): inputData(inputData),numDim(dim), lambda(lambda),biasLambda(biasLambda){

	}


	void setInputData(string file) {
		ifstream fin(file.c_str());
		string s;
		vector<int> uids, pids, qs;
		vector<double> ps;

		while(getline(fin,s)) {
			stringstream ss(s);
			int uid, pid;
			double p, q;
			ss >> uid >> pid >> p >> q;
			uids.push_back(uid);
			pids.push_back(pid);
			ps.push_back(p);
			qs.push_back(q);
		}
		fin.close();

		List inputData = List::create(_["uid"] = uids, _["pid"] = pids, _["price"] = ps, _["quantity"] = qs);
		setInputData(inputData);
	}

	void setInputData(List& inputData) {
		users = as<uvec>(inputData["uid"]);
		items = as<uvec>(inputData["pid"]);
		quantities = as<uvec>(inputData["quantity"]);
		prices = as<vec>(inputData["price"]);
		numUsers = arma::max(users);
		numItems = arma::max(items);
		numRows = users.size();
		summary = List::create(_["num.users"] = numUsers, _["num.items"] = numItems, _["num.rows"] = numRows);

		/// initialize variables
		uMat = mat(numDim, numUsers, fill::randu);
		iMat = mat(numDim, numItems, fill::randu);
		uBias = vec(numUsers,fill::zeros);
		uBias = vec(numItems,fill::zeros);
		globalBias = 0;

		/// get user and item frequences
		userFreq = uvec(numUsers,fill::zeros);
		itemFreq = uvec(numItems, fill::zeros);
		
		for(int i = 0; i < numRows; i++) {
			/// change 1 based to 0 based
			userFreq[users[i] - 1]++;
			itemFreq[items[i] - 1]++;
		}		
	}


	inline double getUserFreqWeight(int idx) {
		return 1/userFreq[idx - 1];
	}

	inline double getItemFreqWeight(int idx) {
		return 1/itemFreq[idx - 1];
	}

	inline double getGlobalFreqWeight() {
		return 1/numRows;
	}


	/*
	* KPR delta consumer surplus
	*/
	inline static double kprDeltaCs(double a, double q, double p) {
		/// % operator for element-wise matrix multiplication
		return a * (log(q+1) - log(q)) - p;
	}

	inline static vec kprDeltaCs(vec& a, vec& q, vec& p) {
		return a % (log(q+1) - log(q)) - p;
	}

	/**
	* sigmoid probability given consumer surplus derivative
	*/
	inline static double lrProb(double deltaCs) {
		return 1 /(1 + exp(-deltaCs));
	}

	/// vectorized version
	inline static vec lrProb(vec& deltaCs) {
		vec ones(deltaCs.size(),fill::ones);
		return ones /(1 + exp(-deltaCs));
	}


	inline static double logLrProb(double x) {
		if (x < -10) {
			return x;
		}
		return log(1/(1 + exp(-x)));
	}

	inline static vec logLrProb(vec& x) {
		vec res(x.size());
		for(int i = 0; i < x.size(); i++) {
			if (x[i] < -10) {
				res[i] = x[i];
			} else {
				res[i] = log(1/(1+exp(-x[i])));
			}
		}
		return res;
	}	

	static int kprPredQuantity(double a, double price) {
		return static_cast<int>(1.0/(exp(price/a) - 1));
	}

	/*
	* sgd 
	*/
	void sgd() {
		int M = numUsers;
		int N = numItems;
		int D = numDim;

		/// sgd settings
		int t = 0;
		int maxEpochs = 50;

		uvec rowIndices(numRows);
		for (int i = 0; i < numRows; i++){
			rowIndices[i] = i;
		}

		double learningRate;

		/// accumuated gradient		
		/// minibatch strategy
		int batchSize = 10;
		std::set<int> batchUsers;
		std::set<int> batchItems;
		mat uMatGrad(D,M, fill::zeros);
		mat iMatGrad(D,N, fill::zeros);
		vec uBiasGrad(M,fill::zeros);
		vec iBiasGrad(N,fill::zeros);
		double globalBiasGrad = 0;

		for (int epoch = 0; epoch < maxEpochs; epoch++) {
			uvec shuffledRowIndices = arma::shuffle(rowIndices);
			double fVal = 0;
			for (int k = 0; k < numRows; k++) {
				int r = shuffledRowIndices[k];
				t += 1;
				int i = users[r] - 1;
				int j = items[r] - 1;
				double p = prices[r];
				double q = quantities[r];

				vec xi = uMat.col(i);
				vec yj = iMat.col(j);
				double bi = uBias[i];
				double bj = iBias[j];
				double aij = evalAij(xi, yj, bi, bj, globalBias);

				double deltaCsQ = kprDeltaCs(aij,q,p);
				double deltaCsQg1 = kprDeltaCs(aij,q + 1, p);

				double probQ = lrProb(deltaCsQ);
				double probQg1 = lrProb(deltaCsQg1);

				/// update function value
				/// negative log likelihood value
				double llVal = -(logLrProb(deltaCsQ) + logLrProb(-deltaCsQg1));
				/// regularization value
				double regVal = 0.5 * (lambda * norm(xi) + lambda * norm(yj) + biasLambda * bi * bi + biasLambda * bj * bj + biasLambda * globalBias * globalBias);
				fVal += (llVal + regVal);

				/// calculate gradient, watch out the sign
				double commonTerm = -( (1 - probQ) * (log(q+1) - log(1)) - probQg1 * (log(q+2) - log(q+1)) );
				/// update the user and item latent vector
				vec deltaXi = yj * commonTerm + xi * lambda;
				uMatGrad.col(i) += deltaXi;
				vec deltaYj = xi * commonTerm + yj * lambda;
				iMatGrad.col(j) += deltaYj;

				/// update bias
				uBiasGrad[i] += (commonTerm + biasLambda * uBias[i]);
				iBiasGrad[j] += (commonTerm + biasLambda * iBias[j]);
				globalBiasGrad += (commonTerm + globalBias);
				batchUsers.insert(i);
				batchItems.insert(j);

				/// update gradient vector
				learningRate = 0.01 /(1 + 1e-4 * t);
				if (t % batchSize == 0 && t > 0) {
					/// only update those affected user and items
					for(set<int>::iterator iter = batchUsers.begin(); iter != batchUsers.end(); ++iter) {
						int tmpUser = *iter;
						uMat.col(tmpUser) -= (uMatGrad.col(tmpUser) * learningRate);
						uMatGrad.col(tmpUser) = 0;	

						uBias[tmpUser] -= (learningRate * uBiasGrad[tmpUser]);
						uBiasGrad[tmpUser] = 0;					
					}

					for(set<int>::iterator iter = batchItems.begin(); iter != batchItems.end(); ++iter) {
						int tmpItem = *iter;
						iMat.col(tmpItem) -= (iMatGrad.col(tmpItem) * learningRate);
						iMatGrad.col(tmpItem) = 0;						

						iBias[tmpItem] -= (learningRate * iBiasGrad[tmpItem]);
						iBiasGrad[tmpItem] = 0;
					}					

					globalBias -= (learningRate * globalBiasGrad);
					globalBiasGrad = 0;

					batchUsers.clear();
					batchItems.clear();
				}
			}
			cout << "epoch:" << epoch << ", function value:" << fVal << endl;
		}
	}


	static double evalAij(vec& xi, vec& yj, double bi, double bj, double globalBias) {
		return arma::accu(xi % yj) + bi + bj + globalBias;
	}

	/// note: user and item are 1 based index, so minus one before using it to access the array
	static vec evalAij(mat& umat, mat& imat, vec& ubias, vec& ibias, double globalBias, uvec& users, uvec& items) {
		/// do by loop 
		/// user and item vector store in column wise
		int n = users.size();
		vec res(n,fill::zeros);
		for (int k = 0; k < n; k++) {
			int uIdx = users[k] - 1;
			int pIdx = items[k] - 1;
			vec xi = umat.col(uIdx);
			vec yj = imat.col(pIdx);
			res[k] = arma::accu(xi % yj) + ubias[uIdx] + ibias[pIdx] + globalBias;
		}
		return res;
	}

};




// [[Rcpp::export]]
List csMax(List& inputData, int dim) {
	CSMax csm(dim);
	csm.setInputData(inputData);
	csm.sgd();
	return csm.summary;
}

// [[Rcpp::export]]
double kprDeltaCs(double a, double q, double p) {
	return CSMax::kprDeltaCs(a,q,p);
}

// [[Rcpp::export]]
vec kprDeltaCsV(vec& a, vec& q, vec& p) {
	return CSMax::kprDeltaCs(a,q,p);
}

// [[Rcpp::export]]
double lrProb(double deltaCs) {
	return CSMax::lrProb(deltaCs);
}


// [[Rcpp::export]]
vec lrProbV(vec& deltaCs) {
	return CSMax::lrProb(deltaCs);
}

// [[Rcpp::export]]
double logLrProb(double x){
	return CSMax::logLrProb(x);
}


// [[Rcpp::export]]
vec logLrProbV(vec& x){
	return CSMax::logLrProb(x);
}



// [[Rcpp::export]]
int kprPredQuantity(double a, double p){
	return CSMax::kprPredQuantity(a,p);
}


/// main funciton
int main(int argc, char** argv) {
	string file = string(argv[1]);
	CSMax csm(10);
	/// provide data from file
	csm.setInputData(file);
	csm.sgd();
	return 0;
}