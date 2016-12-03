#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>
#include <set>
#include <vector>

using namespace arma;
using namespace std;

class Sample {
public:
	int user;
	int item;
	double price;
	double quantity;
	/// utility given by CSMaxSA
	double utility;
	Sample(int user = 0, int item = 0, int price = 0, double quantity = 0, double utility = 0):user(user),item(item),quantity(quantity),utility(utility){}
};


/**
* model parameters
*/
class ModelParam {
public:
	/// product cost percentage
	double costPercentage;
	//// reguarlizer for intermediate quantity \lambda_i,j
	double gamma;
	/// altent dimension
	int dim;
	/// initial learning rate
	double initLearnRate;
	/// max q to evaluate
	int maxQ;
	/// regularizer for user and item latent vectors
	double reg;
	/// bias reg
	double biasReg;

	/// max SGD epochs
	int maxEpochs;
	ModelParam(): costPercentage(0.8), gamma(1),dim(20),initLearnRate(0.1), maxQ(5), reg(0.01),biasReg(0.01), maxEpochs(50)
	{

	}
};

/// output model parameter
ostream& operator << (ostream& oss, const ModelParam& param) {
	oss << "production cost:" << param.costPercentage <<  " ,gamma:" << param.gamma << " ,dim:" << param.dim << " ,lr0:" << param.initLearnRate << " ,maxQ:" << param.maxQ << " ,reg:" << param.reg << " , bias reg:" << param.biasReg << " , max epochs:" << param.maxEpochs;
	return oss;
}


/**
* social surplus maximizer
* the optimizing variables are quantity allocation given user and item
* the quantity is modeled by collaborative filtering
* the basic personalized utility is given by CSMaxSA program and the 
*/
class SSMax {
public:
	/// training samples
	vector<Sample> _trainSamples;
	/// testing samples
	vector<Sample> _testSamples;
	/// stats
	int _numUser;
	int _numItem;
	int _numTrainSamples;
	int _numTestSamples;

	/// # of Samples for each user
	vector<int> _userFreqMap;
	/// # of Samples for each item
	vector<int> _itemFreqMap;
	/// model parameters
	ModelParam _modelParam;

	/// resulteted user and item latent vectors for \lambda_i,j (matrix)
	mat _XM, _YM;
	/// user and item bias values (vector)
	vec _buV, _biV;
	/// global bias (scalar)
	double _b0S;

	// input and output files
	string _trainFile, _testFile, _resultDir;

	double _trainSocialSurplus;
	double _testSocialSurplus;

	/// static varaibles
	static double RECIP_FACT_LOOKUP[];

	/// for CSMax model
	mat _CSM_XM, _CSM_YM;
	vec _CSM_buV, _CSM_biV;
	double _CSM_b0;

	///
	vector<int> _recUsers;
	vector<int> _recItems;

public:
	struct RecItem{
		int item;
		double score;
		RecItem(int item, double score) : item(item),score(score){}

		/// descending order
		bool operator < (const RecItem& rhs) {
			return score > rhs.score;
		}
	};

public:
	SSMax(ModelParam param,string trainFile = "", string testFile  = "", string resultDir = ".", bool randomUtility = false):_numUser(0),_numItem(0),_numTrainSamples(0), _modelParam(param), _trainFile(trainFile),_testFile(testFile),_resultDir(resultDir){
		cout << "------------------------- model parameters -----------------------" << endl;
		cout << _modelParam << endl;
		cout << "------------------------------------------------------------------" << endl;
		/// consumer surplus model
		readInCSModel();
		readTrainingData(randomUtility);
		getUserItemFrequency();
		readTestingData();
	}

	/*
	* model by CSMaxSA application
	* the model is to generate personalized utility given user-item pair
	*
	*/
	void readInCSModel() {
		cout << ">>> read in personalized utility model by CSMaxSA" << endl;
		mat tmpMat;
		tmpMat.load("../data/1015/CSMax_umat.csv", csv_ascii);
		_CSM_XM = tmpMat.t();
		tmpMat.load("../data/1015/CSMax_imat.csv", csv_ascii);
		_CSM_YM = tmpMat.t();
		_CSM_buV.load("../data/1015/CSMax_ubias.csv", csv_ascii);
		_CSM_biV.load("../data/1015/CSMax_ibias.csv", csv_ascii);
		vec b0V;
		b0V.load("../data/1015/CSMax_globalbias.csv", csv_ascii);
		_CSM_b0 = b0V[0];

		/// print model paramters
		cout << "# of users:" << _CSM_XM.n_cols << endl;
		cout << "# of items:" << _CSM_XM.n_cols << endl;
		cout << "# of dimension:" << _CSM_XM.n_rows << endl;

	}

	/**
	* return personalized utility given user-item pair
	*/
	double userItemUtility(int i, int j) {
		return accu(_CSM_XM.col(i) % _CSM_YM.col(j)) + _CSM_buV[i] + _CSM_biV[j] + _CSM_b0;
	}

	/**
	* read in training data samples
	* 
	* each training sample includes user id, item id, product price, purchase quantity and personalized utility by CSMax
	* @param 	randUtility	whether to generate utility randomly
	*/
	void readTrainingData(bool randUtility = false) {
		if(_trainFile != "") {
			cout << ">>> read in training data" << endl;
			ifstream fin(_trainFile.c_str());
			string s;

			while(getline(fin,s)) {
				stringstream ss(s);
				Sample smp;
				ss >> smp.user >> smp.item >> smp.price >> smp.quantity;
				// if (randUtility) {
				// 	/// simply set to price
				// 	smp.utility = smp.price;
				// } else {
				// 	/// else read from the data
				// 	ss >> smp.utility;
				// }
				smp.utility = userItemUtility(smp.user - 1,smp.item - 1);
				if (smp.user > _numUser) {
					_numUser = smp.user;
				}
				if (smp.item > _numItem) {
					_numItem = smp.item;
				}
				_trainSamples.push_back(smp);
			}
			_numTrainSamples = _trainSamples.size();
			fin.close();

			cout << "=========== data summary =============" << endl;
			cout << "number of users:" << _numUser << endl;
			cout << "number of items:" << _numItem << endl;
			cout << "number of training Samples:" << _numTrainSamples << endl; 

			/// initialize variables
			//// user latent vectors, column-wise
			_XM = mat(_modelParam.dim, _numUser, fill::zeros);
			/// item latent vectors, column-wise
			_YM = mat(_modelParam.dim, _numItem, fill::zeros);
			/// user bias
			_buV = vec(_numUser,fill::zeros);
			/// item bias
			_biV = vec(_numItem,fill::zeros);
			/// global bias
			_b0S = 0;			
		} else {
			cerr << "no training data provided" << endl;
		}
	}

	/**
	* read in testing data
	* 
	* each sample is the same to training smaple except the personalized utility might not be provided.
	*/
	void readTestingData() {
		if(_testFile != ""){
			cout << ">>> read in testing data" << endl;
			ifstream fin(_testFile.c_str());
			string s;
			while(getline(fin,s)) {
				stringstream ss(s);
				Sample smp;
				ss >> smp.user >> smp.item >> smp.price >> smp.quantity;
				_testSamples.push_back(smp);
			}
			fin.close();
			_numTestSamples = _testSamples.size();			
		} else {
			cerr << "no testing data is provided" << endl;
		}
	}

	/*
	* user and item frequency in traning dataset
	* 
	* it's simply the reciprocal of the number of occurence for each user and item
	*/
	void getUserItemFrequency() {
		cout << ">>> get user and item frequency" << endl;
		_userFreqMap = vector<int>(_numUser, 0);
		_itemFreqMap = vector<int>(_numItem, 0);

		for(int i = 0; i < _trainSamples.size(); i++ ) {
			Sample & smp = _trainSamples[i];
			_userFreqMap[smp.user - 1]++;
			_itemFreqMap[smp.item - 1]++;
		}
	}

	/*
	* derivative w.r.t \lambda for Poisson distribution
	* @param 	lambda 	Poisson distribution parameter
	* @param 	aij 	personzlied utility by CSMax
	* @param 	cost 	product cost, which is price * _modelParam.costPercentage
	*/
	double poisDistExpectedSSDerivative(double lambda, double aij, double cost) {
		/// quick result for q = 0
		double res = 0;
		for(int q = 1; q <= _modelParam.maxQ; q++) {
			/// poisson part
			double a = poisProbDerivative(lambda, q);
			/// consumer surplus
			double b = aij * log(q + 1) - cost * q;
			res += (a * b);
		}
		return res;
	}


	/*
	* poisson distribution probability 
	*/
	static inline double poisProb(double lambda, int q) {
		return pow(lambda, q) * exp(-lambda) *  RECIP_FACT_LOOKUP[q];
	}

	static inline double poisProbDerivative(double lambda, int q) {
		return ((q - lambda) * exp(-lambda) * pow(lambda,q - 1)) * RECIP_FACT_LOOKUP[q];
	}
	/**
	* expected consumer surplus given quantity is subject to Poisson distribution
	* 
	* @param 	lambda 	Poisson distribution parameter
	* @param 	aij 	Peronslized utiilty by CSMax
	* @param 	cost 	single unit production cost
	*/
	double poisDistExpectedSS(double lambda, double aij, double cost) {
		double res = 0;
		for (int q = 0; q <= _modelParam.maxQ; q++) {
			double a = poisProb(lambda, q);
			/// consumer surplus given utility, quantity and production cost
			double b = aij * log(q + 1) - cost * q;
			res += (a * b);
		}		
		return res;
	}

	/** calulate lambda_i,j
	* @param 	i 	user id
	* @param 	j 	item id
	* @return 	estimated quantity
	*/
	inline double predictQuantity(int i, int j) {
		return accu(_XM.col(i) % _YM.col(j)) + _buV[i] + _biV[j] + _b0S;
	}

	double updateTrainSocialSurplus() {
		double totalSS = 0;
		for (int k = 0; k < _numTrainSamples; k++) {
			Sample& smp = _trainSamples[k];
			int i = smp.user - 1;
			int j = smp.item - 1;
			vec xi = _XM.col(i);
			vec yj = _YM.col(j);
			// extimated Poisson distribution parameter
			double lambdaij = predictQuantity(i,j);
			double price = smp.price;
			double utility = smp.utility;
			/// social surplus for current user-item pair
			double SSij = poisDistExpectedSS(lambdaij, utility, price * _modelParam.costPercentage);
			totalSS += SSij;
		}
		_trainSocialSurplus = totalSS;
		return totalSS;
	}


	double updateTestSocialSurplus() {
		double totalSS = 0;
		for (int k = 0; k < _numTestSamples; k++) {
			Sample& smp = _testSamples[k];
			int i = smp.user - 1;
			int j = smp.item - 1;
			vec xi = _XM.col(i);
			vec yj = _YM.col(j);
			// extimated Poisson distribution parameter
			double lambdaij = predictQuantity(i,j);
			double price = smp.price;
			double utility = smp.utility;
			/// social surplus for current user-item pair
			double SSij = poisDistExpectedSS(lambdaij, utility, price * _modelParam.costPercentage);
			totalSS += SSij;
		}
		_testSocialSurplus = totalSS;
		return totalSS;
	}

	/*
	* sgd 
	*/
	void optimize() {
		/// refer to paper
		int M = _numUser;
		int N = _numItem;
		int D = _modelParam.dim;

		/// sgd settings
		int t = 0;

		/// row indexes
		uvec rowIndices(_numTrainSamples);
		for (int i = 0; i < _numTrainSamples; i++){
			rowIndices[i] = i;
		}

		/// decay with time, lr = lr0/(1+t)
		/// lr0: _modelParam.initLearnRate
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

		// double globalBiasFreqRecip = 1.0 / _numTrainSamples;
		double globalBiasFreqRecip = 1.0;
		/// scale down each term, otherwise the accumulated function value will be nan
		double sampleWeight =  1e-3;

		for (int epoch = 0; epoch < _modelParam.maxEpochs; epoch++) {
			uvec shuffledRowIndices = arma::shuffle(rowIndices);
			/// function value
			double fVal = 0;
			/// total social surplus
			double totalSS = 0;
			double averageLambda = 0;
			for (int k = 0; k < _numTrainSamples; k++) {
				int r = shuffledRowIndices[k];
				/// increase timestamp
				t++;
				Sample& smp = _trainSamples[r];
				int i = smp.user - 1;
				int j = smp.item - 1;
				double p = smp.price;
				/// production cost
				double pCost = p * _modelParam.costPercentage;
				double q = smp.quantity;
				/// utility by CSMax
				double aij = smp.utility;

				vec xi = _XM.col(i);
				vec yj = _YM.col(j);
				double bi = _buV[i];
				double bj = _biV[j];
				double b0 = _b0S;
				/// for regularization term

				// double userFreqRecip = 1.0 / _userFreqMap[i];
				// double itemFreqRecip = 1.0 / _itemFreqMap[j];
				double userFreqRecip = 1.0;
				double itemFreqRecip = 1.0;

				/// evluate Possible distribution parameter: \labmda_{i,j}
				/// standard bias matrix factorization model
				double lambdaij = accu(xi % yj) + bi + bj + b0;
				averageLambda += lambdaij;

				/// Poisson distribution derivative w.r.t \lambda_{i,j} by accumulating over q = 0,1,2,3,4,5
				/// expected social surplus by Poisson distribution
				double expectedSS = poisDistExpectedSS(lambdaij, aij, pCost);
				/// deviation from prior which is given in the data
				double qErr = lambdaij - q;
				/// derivative w.r.t lambda_{i,j} for the Poisson distribution part
				double lambdaDerivPois = poisDistExpectedSSDerivative(lambdaij, aij, pCost);
				/// derivative w.r.t lambda_{i,j} for the regularization part
				double lambdaDerivReg = -_modelParam.gamma * qErr;
				/// update function value, watch out the minu sign, the objective function is minimization
				fVal += sampleWeight * (-expectedSS + 0.5 * _modelParam.gamma * qErr * qErr + _modelParam.reg * (userFreqRecip * norm(xi) + itemFreqRecip * norm(yj)) + _modelParam.biasReg * (userFreqRecip * bi + itemFreqRecip * bj + globalBiasFreqRecip * b0));
				/// update the user and item latent vector
				/// watch out the minus sign
				double commonTerm = -(lambdaDerivPois + lambdaDerivReg);
				vec deltaXi = sampleWeight * (commonTerm * yj + _modelParam.reg * userFreqRecip * xi);
				vec deltaYj = sampleWeight * (commonTerm * xi + _modelParam.reg * itemFreqRecip * yj);
				uMatGrad.col(i) += deltaXi;
				iMatGrad.col(j) += deltaYj;
				/// update bias
				uBiasGrad[i] += sampleWeight * (commonTerm + _modelParam.biasReg * userFreqRecip * _buV[i]);
				iBiasGrad[j] += sampleWeight * (commonTerm + _modelParam.biasReg * itemFreqRecip * _biV[j]);
				globalBiasGrad += sampleWeight * (commonTerm + _modelParam.biasReg * globalBiasFreqRecip * b0);
				/// keep track of users and items in current batch
				batchUsers.insert(i);
				batchItems.insert(j);

				/// update gradient vector
				learningRate = _modelParam.initLearnRate/(1 + epoch * 0.01);

				/// apply the aggregated gradient
				if ((t % batchSize == 0 && t > 0) || t % _numTrainSamples == 0) {
					/// only update those affected user and items
					for(set<int>::iterator iter = batchUsers.begin(); iter != batchUsers.end(); ++iter) {
						int tmpUser = *iter;
						/// descent
						_XM.col(tmpUser) -= (uMatGrad.col(tmpUser) * learningRate);
						/// reset gradient vector of the user
						uMatGrad.col(tmpUser) = vec(D,fill::zeros);	

						_buV[tmpUser] -= (learningRate * uBiasGrad[tmpUser]);
						uBiasGrad[tmpUser] = 0;					
					}

					for(set<int>::iterator iter = batchItems.begin(); iter != batchItems.end(); ++iter) {
						int tmpItem = *iter;
						_YM.col(tmpItem) -= (iMatGrad.col(tmpItem) * learningRate);
						iMatGrad.col(tmpItem) = vec(D,fill::zeros);						

						_biV[tmpItem] -= (learningRate * iBiasGrad[tmpItem]);
						iBiasGrad[tmpItem] = 0;
					}					

					_b0S -= (learningRate * globalBiasGrad);
					globalBiasGrad = 0;

					batchUsers.clear();
					batchItems.clear();
				}
			}
			averageLambda /= _numTrainSamples;
			double trainRmse = updateTrainRmse();
			double testRmse = updateTestRmse();
			/// update train social surplus
			updateTrainSocialSurplus();
			cout << "epoch:" << epoch << ", average lambda:" << averageLambda << " ,function value:" << fVal << ", learning rate:" << learningRate << " ,training rmse:" << trainRmse << " ,train social surplus:" << _trainSocialSurplus <<  " ,test rmse:" << testRmse << endl;
		}
		double rmse = updateTrainRmse();
		double rmse1 = updateTestRmse();
		cout << "final training rmse:" << rmse  << " ,test rmse:" << rmse1 << endl;
	}


	void loadRecommendUserItems() {
		/// read in users
		ifstream fin("../data/1015/rec_users.txt");
		string line;
		_recUsers.clear();
		while (getline(fin, line)) {
			int user;
			stringstream ss(line);
			ss >> user;
			_recUsers.push_back(user - 1);
		}
		fin.close();
		cout << ">>> # of rec users:" << _recUsers.size() << endl;
		
		fin.open("../data/1015/rec_items.txt");
		while (getline(fin, line)) {
			int item;
			stringstream ss(line);
			ss >> item;
			_recItems.push_back(item - 1);
		}
		fin.close();
		cout << ">>> # of rec items:" << _recItems.size() << endl;
	}


	/*
	* model by SSMax application
	* the model is to generate personalized utility given user-item pair
	*
	*/
	void loadSSModel() {
		cout << ">>> read in social surplus maximization model by SSMax" << endl;
		mat tmpMat;
		tmpMat.load("../data/1015/SSMax_umat.csv", csv_ascii);
		_XM = tmpMat.t();
		tmpMat.load("../data/1015/SSMax_imat.csv", csv_ascii);
		_YM = tmpMat.t();
		_buV.load("../data/1015/SSMax_ubias.csv", csv_ascii);
		_biV.load("../data/1015/SSMax_ibias.csv", csv_ascii);
		vec b0V;
		b0V.load("../data/1015/SSMax_globalbias.csv", csv_ascii);
		_b0S = b0V[0];
		/// print model paramters
		cout << "# of users:" << _XM.n_cols << endl;
		cout << "# of items:" << _YM.n_cols << endl;
		cout << "# of dimension:" << _XM.n_rows << endl;
	}

	void recommend() {
		loadRecommendUserItems();
		/// do it for each user
		cout << ">>> generate recommendations" << endl;
		/// include model parameters for each result
		stringstream fnss;
		fnss << _resultDir << "/SSMAX_gamma" << _modelParam.gamma << "_rec_useritem_ranked_top100.csv";
		/// pass to output
		ofstream ofs(fnss.str().c_str());
		for(auto & user: _recUsers) {
			vector<RecItem> itemScores;
			for(auto & item : _recItems) {
				double qij = predictQuantity(user,item);
				itemScores.push_back(RecItem(item, qij));
			}
			sort(itemScores.begin(),itemScores.end());
			/// output top 100 result
			for (int i = 0; i < 100; i++) {
				RecItem& rec = itemScores[i];
				ofs << user + 1 << "\t" << rec.item + 1<< "\t" << rec.score << endl;
			}
		}
		ofs.close();
	}

	void saveModel() {
		cout << "save model parameters" << endl;
		stringstream ss;
		mat umt = _XM.t();
		ss << _resultDir << "/SSMAX_gamma" << _modelParam.gamma << "_umat.csv";
		umt.save(ss.str(),csv_ascii);
		mat imt = _YM.t();

		stringstream().swap(ss);
		ss << _resultDir << "/SSMAX_gamma" << _modelParam.gamma << "_imat.csv";
		imt.save(ss.str(), csv_ascii);

		stringstream().swap(ss);
		ss << _resultDir << "/SSMAX_gamma" << _modelParam.gamma << "_ubias.csv";		
		_buV.save(ss.str(),csv_ascii);

		stringstream().swap(ss);
		ss << _resultDir << "/SSMAX_gamma" << _modelParam.gamma << "_ibias.csv";				
		_biV.save(ss.str(),csv_ascii);
		vec b0v(1);
		b0v[0] = _b0S;

		stringstream().swap(ss);
		ss << _resultDir << "/SSMAX_gamma" << _modelParam.gamma << "_globalbias.csv";				
		b0v.save(ss.str(),csv_ascii);
	}


	/// training rmse
	double updateTrainRmse() {
		double rmse = 0;
		for(int k = 0; k < _numTrainSamples; k++) {
			Sample& smp = _trainSamples[k];
			int i = smp.user - 1;
			int j = smp.item - 1;
			int q = smp.quantity;
			double p = smp.price;
			double aij = smp.utility;
			double predQ = predictQuantity(i,j);
			double err = predQ - q;
			rmse += (err * err);
		}
		// cout << "training inf cnt:" << infCnt << endl;
		return sqrt(rmse / _numTrainSamples);
	}

	/// training rmse
	double updateTestRmse() {
		double rmse = 0;
		for(int k = 0; k < _numTestSamples; k++) {
			Sample& smp = _testSamples[k];
			int i = smp.user - 1;
			int j = smp.item - 1;
			int q = smp.quantity;
			double predQ = predictQuantity(i,j);
			double err = predQ - q;
			rmse += (err * err);
		}
		return sqrt(rmse / _numTestSamples);
	}

};

/// RECIP_FACT_LOOKUP for 0,1,2,3,4,5,6,7,8,9,10
double SSMax::RECIP_FACT_LOOKUP[] = {1.000000e+00, 1.000000e+00, 5.000000e-01, 1.666667e-01, 4.166667e-02,8.333333e-03, 1.388889e-03, 1.984127e-04, 2.480159e-05, 2.755732e-06, 2.755732e-07, 2.505211e-08, 2.087676e-09, 1.605904e-10, 1.147075e-11, 7.647164e-13, 4.779477e-14, 2.811457e-15, 1.561921e-16, 8.220635e-18, 4.110318e-19};

/// main funciton

/// run social surplus optimizer
#define __OPTIMIZE_SS__

/// load existing model and run recommendation
// #dfine __RECOMMEND__

int main(int argc, char** argv) {
	if (argc != 7) {
		cout << "usage: SSMAX <train data file> <test data file> <result dir> <gamma> <max epochs> <cost percentage>" << endl;
		return 0;
	}

	ModelParam modelParam;
	modelParam.dim = 20;
	/// really large value so that the reuslt will be exactly CF on quantity
	// modelParam.gamma = 5;
	modelParam.costPercentage = 0.5;
	modelParam.reg = 0.05;
	modelParam.biasReg = 0.05;
	modelParam.maxQ = 20;
	modelParam.initLearnRate = 0.1;
	string trainFile = string(argv[1]);
	string testFile = string(argv[2]);
	string resultDir = string(argv[3]);
	/// quantity regularizer
	modelParam.gamma = stod(string(argv[4]));
	modelParam.maxEpochs = stoi(string(argv[5]));
	modelParam.costPercentage = stod(string(argv[6]));

	/// randomly generate utlility for testing purpose
	bool randomUtility = true;
#ifdef __OPTIMIZE_SS__
	SSMax ssm(modelParam, trainFile, testFile, resultDir, randomUtility);
	ssm.optimize();
	ssm.saveModel();
	ssm.recommend();
#endif

#ifdef __RECOMMEND__
	trainFile = "";
	testFile = "";
	SSMax ssm(modelParam,trainFile,testFile,resultDir);
	ssm.loadSSModel();
	ssm.recommend();
#endif
	return 0;
}