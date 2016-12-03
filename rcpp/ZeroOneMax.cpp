#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>
#include <set>
#include <vector>

using namespace arma;
using namespace std;

/**
* maximize single unit utility
* quantity > 0: U_i,j(q=1) = a_i,j - p_j > 0
* quantity = 0: U_i,j(q=0) = a_i,j - p_j < 0
*/
class ZeroOneMax {
public:
	vector<int> users, items;
	vector<double> quantities;
	vector<double> prices;

	vector<int> testUsers, testItems;
	vector<double> testPrices,testQuantities;

	int numDim;
	int numUsers;
	int numItems;
	int numRows;

	mat uMat, iMat;
	vec uBias, iBias;
	double globalBias;

	/// hyper parameters
	double lambda, biasLambda;

	static double UTILITY_SCALE;
public:
	ZeroOneMax(int dim = 5, double lambda = 0.01, double biasLambda = 0.01): numDim(dim), lambda(lambda),biasLambda(biasLambda){

	}


	inline static double logSigProb(double x) {
		if (x < -10) {
			return x;
		}
		return log(1/(1 + exp(-x)));
	}

	void readTrainingData(string file) {
		ifstream fin(file.c_str());
		string s;
		while(getline(fin,s)) {
			stringstream ss(s);
			int uid, pid;
			double p,q;
			ss >> uid >> pid >> p >> q;
			p = 0;
			/// change to binary flag
			q = (q > 0? 1 : -1);
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
			int uid, pid;
			double p,q;
			ss >> uid >> pid >> p >> q;
			p = 0;
			q = (q > 0 ? 1 : -1);
			testUsers.push_back(uid);
			testItems.push_back(pid);
			testPrices.push_back(p);
			testQuantities.push_back(q);
		}
		fin.close();
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

				/// update function value
				/// negative log sigmoid function
				double sigProb = 1/(1 + exp(-q * (aij - p)));
				double llVal = -logSigProb(q * (aij - p));
				/// regularization value
				double regVal = 0.5 * (lambda * norm(xi) + lambda * norm(yj) + biasLambda * bi * bi + biasLambda * bj * bj + biasLambda * globalBias * globalBias);
				fVal += (llVal + regVal);

				/// calculate gradient, watch out the sign
				double commonTerm = -q * (1 - sigProb);
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
				learningRate = 0.1;
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
			double trainMatchRatio = 0;
			double testMatchRatio = 0;
			double rmse = trainRmse(trainMatchRatio);
			double rmse1 = testRmse(testMatchRatio);
			cout << "epoch:" << epoch << ", function value:" << fVal << ", learning rate:" << learningRate << " ,training rmse:" << rmse << " ,train match:" << trainMatchRatio << " ,test rmse:" << rmse1 << ", test match:" << testMatchRatio  << endl;
		}
		double matchRatio;
		double testMatchRatio;
		double rmse = trainRmse(matchRatio);
		double rmse1 = testRmse(testMatchRatio);
		cout << "final training rmse:" << rmse  << " ,test rmse:" << rmse1 << endl;
	}


	void saveModel() {
		cout << "save model parameters" << endl;
		mat umt = uMat.t();
		umt.save("../data/sgd/withneg_ss_umat.csv",csv_ascii);
		mat imt = iMat.t();
		imt.save("../data/sgd/withneg_ss_imat.csv", csv_ascii);
		uBias.save("../data/sgd/withneg_ss_ubias.csv",csv_ascii);
		iBias.save("../data/sgd/withneg_ss_ibias.csv",csv_ascii);
		vec b0v(1);
		b0v[0] = globalBias;
		b0v.save("../data/sgd/withneg_ss_globalbias.csv",csv_ascii);
	}

	static double userItemUtility(vec& xi, vec& yj, double bi, double bj, double b0) {
		return UTILITY_SCALE * (arma::accu(xi % yj) + bi + bj + b0);
	}

	/// training rmse
	double trainRmse(double& matchRatio) {
		double rmse = 0;
		int matchCnt = 0;
		for(int k = 0; k < numRows; k++) {
			int i = users[k] - 1;
			int j = items[k] - 1;
			double q = quantities[k];
			double p = prices[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double aij = userItemUtility(xi,yj,uBias[i],iBias[j],globalBias);
			// probability of purchasing 1 unit
			double sigProb = 1/(1 + exp(-q * (aij - p)));
			double err = sigProb > 0.5? 0 : 1;
			rmse += err;
		}
		return rmse / numRows;
	}

	void saveTestPrediction(){
		ofstream ofs("../data/sgd/withneg_ss_test_prediction.csv");
		int numTestings = testUsers.size();
		for(int k = 0; k < numTestings; k++) {
			int i = testUsers[k] - 1;
			int j = testItems[k] - 1;
			double q = testQuantities[k];
			double p = testPrices[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double aij = userItemUtility(xi,yj,uBias[i],iBias[j],globalBias);
			double sigProb = 1/(1 + exp(-(aij - p)));
			ofs << (i + 1) << "\t" << (j + 1) <<"\t" << p << "\t" << q << "\t" << sigProb << endl;
			/// also check inequation
		}
		ofs.close();
	}

	/// training rmse
	double testRmse(double& matchRatio) {
		double rmse = 0;
		int numRows = testUsers.size();
		for(int k = 0; k < numRows; k++) {
			int i = testUsers[k] - 1;
			int j = testItems[k] - 1;
			double q = testQuantities[k];
			double p = testPrices[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double aij = userItemUtility(xi,yj,uBias[i],iBias[j],globalBias);
			double sigProb = 1/(1 + exp(-q * (aij - p)));
			double err = (sigProb > 0.5 ? 0 : 1);
			rmse += err;	
		}
		return rmse / numRows;
	}

};


double ZeroOneMax::UTILITY_SCALE = 1;

/// main funciton

int main(int argc, char** argv) {
	string trainFile = string(argv[1]);
	string testFile = string(argv[2]);
	int dim = 20;
	double lambda = 0.02;
	double biasLambda = lambda * 0.5;
	int maxEpochs = 50;
	ZeroOneMax csm(dim,lambda,biasLambda);
	/// provide data from file
	csm.readTrainingData(trainFile);
	csm.readTestingData(testFile);
	csm.sgd(maxEpochs);
	cout << "sgd parameters: dim - " << dim << ", lambda - " << lambda << " bias lambda - ," << biasLambda << ", max epochs - " << maxEpochs << endl;
	csm.saveModel();
	csm.saveTestPrediction();
	return 0;
}