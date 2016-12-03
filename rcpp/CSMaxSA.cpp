#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>
#include <set>
#include <vector>

using namespace arma;
using namespace std;

/**
* consumer surplus maximization through KPR Utility model
*/
class CSMax {
public:
	vector<int> users, items, quantities;
	vector<double> prices;

	vector<int> testUsers, testItems, testQuantities;
	vector<double> testPrices;

	int numDim;
	int numUsers;
	int numItems;
	int numRows;

	mat uMat, iMat;
	vec uBias, iBias;
	double globalBias;

	/// hyper parameters
	double lambda, biasLambda;

public:
	CSMax(int dim = 5, double lambda = 0.01, double biasLambda = 0.01): numDim(dim), lambda(lambda),biasLambda(biasLambda){

	}


	void readTrainingData(string file) {
		ifstream fin(file.c_str());
		string s;
		while(getline(fin,s)) {
			stringstream ss(s);
			int uid, pid;
			double p, q;
			ss >> uid >> pid >> p >> q;
			users.push_back(uid);
			items.push_back(pid);
			prices.push_back(p);
			quantities.push_back(q);
			// cout << uid << "\t" << pid << "\t" << p << "\t" << q << endl;
		}
		fin.close();
		numUsers = *(std::max_element(users.begin(),users.end()));
		numItems = *(std::max_element(items.begin(),items.end()));
		numRows = users.size();

		cout << "=========== data summary =============" << endl;
		cout << "number of users:" << numUsers << endl;
		cout << "number of items:" << numItems << endl;
		cout << "number of rows:" << numRows << endl; 

		/// initialize variables
		uMat = mat(numDim, numUsers, fill::randu);
		iMat = mat(numDim, numItems, fill::randu);
		uBias = vec(numUsers,fill::zeros);
		iBias = vec(numItems,fill::zeros);
		globalBias = 0;		
	}


	void readTestingData(string file) {
		ifstream fin(file.c_str());
		string s;
		while(getline(fin,s)) {
			stringstream ss(s);
			int uid, pid,q;
			double p;
			ss >> uid >> pid >> p >> q;
			testUsers.push_back(uid);
			testItems.push_back(pid);
			testPrices.push_back(p);
			testQuantities.push_back(q);
		}
		fin.close();
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

	/*
	* sgd 
	*/
	void sgd(int maxEpochs = 50) {
		int M = numUsers;
		int N = numItems;
		int D = numDim;

		/// sgd settings
		int t = 0;

		uvec rowIndices(numRows);
		for (int i = 0; i < numRows; i++){
			rowIndices[i] = i;
		}

		double learningRate;

		/// accumuated gradient		
		/// minibatch strategy
		int batchSize = 20;
		set<int> batchUsers;
		set<int> batchItems;
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
				/// increase timestamp
				t++;
				int i = users[r] - 1;
				int j = items[r] - 1;
				double p = prices[r];
				double q = quantities[r];

				vec xi = uMat.col(i);
				vec yj = iMat.col(j);
				double bi = uBias[i];
				double bj = iBias[j];
				double aij = userItemUtility(xi, yj, bi, bj, globalBias);

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
				double commonTerm = -( (1 - probQ) * (log(q+1) - log(q)) - probQg1 * (log(q+2) - log(q+1)) );
				/// update the user and item latent vector
				vec deltaXi = yj * commonTerm + xi * lambda;
				uMatGrad.col(i) += deltaXi;
				vec deltaYj = xi * commonTerm + yj * lambda;
				iMatGrad.col(j) += deltaYj;

				/// update bias
				uBiasGrad[i] += (commonTerm + biasLambda * uBias[i]);
				iBiasGrad[j] += (commonTerm + biasLambda * iBias[j]);
				globalBiasGrad += (commonTerm + biasLambda * globalBias);
				batchUsers.insert(i);
				batchItems.insert(j);

				/// update gradient vector
				learningRate = 0.1/(1 + epoch * 0.1);
				/// apply the aggregated gradient
				if ((t % batchSize == 0 && t > 0) || t % numRows == 0) {
					/// only update those affected user and items
					for(set<int>::iterator iter = batchUsers.begin(); iter != batchUsers.end(); ++iter) {
						int tmpUser = *iter;
						/// descent
						uMat.col(tmpUser) -= (uMatGrad.col(tmpUser) * learningRate);
						/// reset gradient vector of the user
						uMatGrad.col(tmpUser) = vec(D,fill::zeros);	

						uBias[tmpUser] -= (learningRate * uBiasGrad[tmpUser]);
						uBiasGrad[tmpUser] = 0;					
					}

					for(set<int>::iterator iter = batchItems.begin(); iter != batchItems.end(); ++iter) {
						int tmpItem = *iter;
						iMat.col(tmpItem) -= (iMatGrad.col(tmpItem) * learningRate);
						iMatGrad.col(tmpItem) = vec(D,fill::zeros);						

						iBias[tmpItem] -= (learningRate * iBiasGrad[tmpItem]);
						iBiasGrad[tmpItem] = 0;
					}					

					globalBias -= (learningRate * globalBiasGrad);
					globalBiasGrad = 0;

					batchUsers.clear();
					batchItems.clear();
				}
			}
			double rmse = trainRmse();
			double rmse1 = testRmse();
			cout << "epoch:" << epoch << ", function value:" << fVal << ", learning rate:" << learningRate << " ,training rmse:" << rmse << " ,test rmse:" << rmse1 << endl;
		}
		double rmse = trainRmse();
		double rmse1 = testRmse();
		cout << "final training rmse:" << rmse  << " ,test rmse:" << rmse1 << endl;
	}


	void saveModel() {
		cout << "save model parameters" << endl;
		mat umt = uMat.t();
		umt.save("../data/1015/CSMax_umat.csv",csv_ascii);
		mat imt = iMat.t();
		imt.save("../data/1015/CSMax_imat.csv", csv_ascii);
		uBias.save("../data/1015/CSMax_ubias.csv",csv_ascii);
		iBias.save("../data/1015/CSMax_ibias.csv",csv_ascii);
		vec b0v(1);
		b0v[0] = globalBias;
		b0v.save("../data/1015/CSMax_globalbias.csv",csv_ascii);
	}

	static double userItemUtility(vec& xi, vec& yj, double bi, double bj, double b0) {
		return arma::accu(xi % yj) + bi + bj + b0;
	}

	/// note: user and item are 1 based index, so minus one before using it to access the array
	static vec userItemUtility(mat& umat, mat& imat, vec& ubias, vec& ibias, double globalBias, uvec& users, uvec& items) {
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

	/// training rmse
	double trainRmse() {
		double rmse = 0;
		int infCnt = 0;
		for(int k = 0; k < numRows; k++) {
			int i = users[k] - 1;
			int j = items[k] - 1;
			int q = quantities[k];
			double p = prices[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double aij = userItemUtility(xi,yj,uBias[i],iBias[j],globalBias);
			double predQ = kprPredQuantity(aij,p);
			if (isinf(predQ)) {
				infCnt++;
			} else {
				rmse += ((predQ - q) * (predQ - q));
			}
		}
		// cout << "training inf cnt:" << infCnt << endl;

		return sqrt(rmse/(numRows - infCnt));
	}

	void saveTestPrediction(){
		ofstream ofs("../data/1015/CSMax_test_prediction.csv");
		int numTestings = testUsers.size();
		int signMatch = 0;
		int infCnt = 0;
		for(int k = 0; k < numTestings; k++) {
			int i = testUsers[k] - 1;
			int j = testItems[k] - 1;
			int q = testQuantities[k];
			double p = testPrices[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double aij = userItemUtility(xi,yj,uBias[i],iBias[j],globalBias);
			double predQ = kprPredQuantity(aij,p);
			ofs << (i + 1) << "\t" << (j + 1) <<"\t" << p << "\t" << q << "\t" << predQ << endl;
			/// also check inequation
			if (!isinf(predQ)) {
				int intPredQ = (int)predQ;
				double deltaCsQ = kprDeltaCs(aij,intPredQ, p);
				double deltaCsQg1 = kprDeltaCs(aij, intPredQ + 1, p);
				/// check sign
				if (deltaCsQ > 0 && deltaCsQg1 < 0) signMatch++;
			} else {
				infCnt++;
			}
		}
		ofs.close();

		cout << signMatch << " out of " << (numTestings - infCnt) << " match" << endl;
	}

	/// training rmse
	double testRmse() {
		double rmse = 0;
		int numTestings = testUsers.size();
		int infCnt = 0;
		for(int k = 0; k < numTestings; k++) {
			int i = testUsers[k] - 1;
			int j = testItems[k] - 1;
			int q = testQuantities[k];
			double p = testPrices[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double aij = userItemUtility(xi,yj,uBias[i],iBias[j],globalBias);
			double predQ = kprPredQuantity(aij,p);
			if (isinf(predQ)) {
				infCnt++;
			} else {
				rmse += ((predQ - q) * (predQ - q));
			}
		}
		// cout << "testing inf cnt:" << infCnt << endl;
		return sqrt(rmse/(numTestings - infCnt));
	}

	/// predict q given utility a and price
	static double kprPredQuantity(double a, double price) {
		double tmp = exp(price/a) - 1;
		return (1.0/tmp);
	}

};

/// main funciton
int main(int argc, char** argv) {
	string trainFile = string(argv[1]);
	string testFile = string(argv[2]);
	int dim = 20;
	double lambda = 0.05;
	double biasLambda = lambda * 0.2;
	int maxEpochs = 100;
	CSMax csm(dim,lambda,biasLambda);
	/// provide data from file
	csm.readTrainingData(trainFile);
	csm.readTestingData(testFile);
	csm.sgd(maxEpochs);
	cout << "sgd parameters: dim - " << dim << ", lambda - " << lambda << " bias lambda - ," << biasLambda << ", max epochs - " << maxEpochs << endl;
	csm.saveModel();
	csm.saveTestPrediction();
	return 0;
}