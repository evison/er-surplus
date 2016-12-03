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
class CF {
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
	CF(int dim = 5, double lambda = 0.01, double biasLambda = 0.01): numDim(dim), lambda(lambda),biasLambda(biasLambda){

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
			// if (q > 0) q = 1;
			quantities.push_back(q);
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
			// only clamp to one when modeling purchasing probability, but this can be done through ZeroOneMax
			// if (q > 0) q = 1;
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
				double q = quantities[r];

				vec xi = uMat.col(i);
				vec yj = iMat.col(j);
				double bi = uBias[i];
				double bj = iBias[j];
				/// predicted quantity
				double predQ = accu(xi % yj) + bi  + bj + globalBias;
				double err = predQ - q;
				/// update function value
				fVal += (0.5 * err * err + 0.5 * (lambda * norm(xi) + lambda * norm(yj) + biasLambda * bi * bi + biasLambda * bj * bj + biasLambda * globalBias * globalBias));
				/// negative log likelihood value
				/// calculate gradient, watch out the sign
				/// update the user and item latent vector
				vec deltaXi = yj * err + xi * lambda;
				uMatGrad.col(i) += deltaXi;
				vec deltaYj = xi * err + yj * lambda;
				iMatGrad.col(j) += deltaYj;

				/// update bias
				uBiasGrad[i] += (err + biasLambda * uBias[i]);
				iBiasGrad[j] += (err + biasLambda * iBias[j]);
				globalBiasGrad += (err + biasLambda * globalBias);
				batchUsers.insert(i);
				batchItems.insert(j);

				/// update gradient vector
				learningRate = 0.05/(1 + epoch * 0.01);
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
		umt.save("../data/1015/quantity_cf_umat.csv",csv_ascii);
		mat imt = iMat.t();
		imt.save("../data/1015/quantity_cf_imat.csv", csv_ascii);
		uBias.save("../data/1015/quantity_cf_ubias.csv",csv_ascii);
		iBias.save("../data/1015/quantity_cf_ibias.csv",csv_ascii);
		vec b0v(1);
		b0v[0] = globalBias;
		b0v.save("../data/1015/quantity_cf_globalbias.csv",csv_ascii);
	}

	/// training rmse
	double trainRmse(double& matchRatio) {
		double rmse = 0;
		int matchCnt = 0;
		for(int k = 0; k < numRows; k++) {
			int i = users[k] - 1;
			int j = items[k] - 1;
			int q = quantities[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double predQ = accu(xi % yj) + uBias[i] + iBias[j] + globalBias;
			rmse += ((predQ - q) * (predQ - q));
		}
		// cout << "training inf cnt:" << infCnt << endl;
		matchRatio = (double)matchCnt / numRows;
		return sqrt(rmse/(numRows));
	}

	void saveTestPrediction(){
		ofstream ofs("../data/1015/quantity_cf_test_prediction.csv");
		int numTestings = testUsers.size();
		for(int k = 0; k < numTestings; k++) {
			int i = testUsers[k] - 1;
			int j = testItems[k] - 1;
			int q = testQuantities[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double predQ = accu(xi % yj) + uBias[i] + iBias[j] + globalBias;
			ofs << (i + 1) << "\t" << (j + 1) << "\t" << q << "\t" << predQ << endl;
			/// also check inequation
		}
		ofs.close();
	}

	void generateRecommendation(){
		ifstream ifs("../data/1015/rec_useritem.csv");
		ofstream ofs("../data/1015/quantity_cf_rec_useritem_ranked.csv");
		string s;
		while(getline(ifs,s)) {
			stringstream ss(s);
			int uid, pid;
			ss >> uid >> pid;
			int i = uid - 1;
			int j = pid - 1;
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double predQ = accu(xi % yj) + uBias[i] + iBias[j] + globalBias;
			ofs << (i + 1) << "\t" << (j + 1) << "\t" << predQ << endl;
		}
		ifs.close();
		ofs.close();
	}


	/// training rmse
	double testRmse(double& matchRatio) {
		double rmse = 0;
		int numTestings = testUsers.size();
		int matchCnt = 0;
		for(int k = 0; k < numTestings; k++) {
			int i = testUsers[k] - 1;
			int j = testItems[k] - 1;
			int q = testQuantities[k];
			vec xi = uMat.col(i);
			vec yj = iMat.col(j);
			double predQ = accu(xi % yj) + uBias[i] + iBias[j] + globalBias;
			rmse += ((predQ - q) * (predQ - q));		
		}
		// cout << "testing inf cnt:" << infCnt << endl;
		matchRatio = (double)matchCnt / numTestings;
		return sqrt(rmse/numTestings);
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
	double lambda = 0.1;
	double biasLambda = 0.01;
	int maxEpochs = 100;
	CF cf(dim,lambda,biasLambda);
	/// provide data from file
	cf.readTrainingData(trainFile);
	cf.readTestingData(testFile);
	cf.sgd(maxEpochs);
	cout << "sgd parameters: dim - " << dim << ", lambda - " << lambda << " bias lambda - ," << biasLambda << ", max epochs - " << maxEpochs << endl;
	cf.saveModel();
	cf.saveTestPrediction();
	cf.generateRecommendation();
	return 0;
}