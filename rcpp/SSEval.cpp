// social surplus evaluation given user-item pairs

#include <iostream>
#include <string>
#include <fstream>
#include <armadillo>
#include <set>
#include <vector>
#include <unordered_map>

using namespace arma;
using namespace std;

class SSEval {
public:
	/// for CSMax model
	mat _CS_XM, _CS_YM;
	vec _CS_buV, _CS_biV;
	double _CS_b0S;

	/// for SSMax model	
	mat _SS_XM, _SS_YM;
	vec _SS_buV, _SS_biV;
	double _SS_b0S;

	/// training file which contains price information
	string _trainFile;
	string _ssModelFilePrefix;
	/// prefix of recommendation list, the full path should be prefix_rec_useritem_ranked_top100.csv
	string _recListFilePrefix;
	unordered_map<int, vector<double> > _productPrices;
	unordered_map<int, double> _productPrice;
	/// cost percentage
	double _costPercent;

	double _totalSS;
	SSEval(string trainFile, string ssModelFilePrefix, string recListFile, double _costPercent) : _trainFile(trainFile), _ssModelFilePrefix(ssModelFilePrefix), _recListFilePrefix(recListFile),_costPercent(_costPercent), _totalSS(0){
		loadSSModel();
		loadCSModel();
		loadPriceInfo();
	}

	/*
	* load product price information
	*/
	void loadPriceInfo() {
		cout << ">>> load product information from train file: " << _trainFile << endl;
		ifstream fin(_trainFile);
		if(fin.good()) {
			string line;
			while(getline(fin,line)) {
				stringstream ss(line);
				int user, item, price, quantity;
				ss >> user >> item >> price >> quantity;
				_productPrices[user].push_back(price);
			}
			/// use the average price for each product
			for(auto& kv : _productPrices) {
				int itemId = kv.first;
				vector<double>& prices = kv.second;
				double avgPrice = 0;
				for(auto & p : prices) {
					avgPrice += p;
				}
				avgPrice /= prices.size();
				_productPrice[itemId] = avgPrice;
			}
		} else {
			cerr << "failed to open training file: " << _trainFile << endl;
		}
	}

	/*
	* compute user-item social surplus by looking SSMax model and CSMax model
	* @param 	i 	user index (0 based)
	* @param 	j 	item index (0 based)
	* @return 	social surplus
	*/
	double userItemSS(int i, int j) {
		vec xi = _SS_XM.col(i);
		vec yj = _SS_YM.col(j);
		double qij = accu(xi % yj) + _SS_buV[i] + _SS_biV[j] + _SS_b0S;

		xi = _CS_XM.col(i);
		yj = _CS_YM.col(j);
		double aij = accu(xi % yj) + _CS_buV[i] + _CS_biV[j] + _CS_b0S;
		double p = _productPrice[j];

		return aij * log(qij + 1) - qij * p * _costPercent;
	}

	/**
	* total social surplus
	*/
	void sumRecSS() {
		cout << ">>> accumulate social surplus of the recommendations" << endl;
		string recListFile = _recListFilePrefix + "_rec_useritem_ranked_top100.csv";
		cout << ">>> load recommendation result from: " << recListFile << endl;
		ifstream fin(recListFile);
		/// save result at each of top 100 positions
		ofstream fout(_recListFilePrefix + "_sssum_top100.csv");
		int lineCounter = 0;
		/// ss for single user at diffent list position
		double userSumSS = 0;
		if(fin.good()) {
			string line;
			while(getline(fin,line)) {
				int user,item;
				double score;
				stringstream ss(line);
				ss >> user >> item >> score;
				double uiSS = userItemSS(user - 1, item - 1);
				/// clamp it to zero is SS is negative
				if (uiSS < 0) uiSS = 0;
				userSumSS += uiSS;
				fout << (lineCounter > 0 ? "," : "") << userSumSS;
				lineCounter++;
				/// reset line counter
				if (lineCounter == 100) {
					fout << endl;
					lineCounter = 0;
					/// also reset userSumSS
					userSumSS = 0;
				}
				///
				_totalSS += uiSS;
			}
		} else {
			cerr << "failed to open recommendation list" << endl;
		}
		fout.close();
		fin.close();
	}


	/*
	* model by CSMaxSA application
	* the model is to generate personalized utility given user-item pair
	*
	*/
	void loadCSModel() {
		cout << ">>> read in personalized utility model by CSMaxSA" << endl;
		mat tmpMat;
		tmpMat.load("../data/1015/CSMax_umat.csv", csv_ascii);
		_CS_XM = tmpMat.t();
		tmpMat.load("../data/1015/CSMax_imat.csv", csv_ascii);
		_CS_YM = tmpMat.t();
		_CS_buV.load("../data/1015/CSMax_ubias.csv", csv_ascii);
		_CS_biV.load("../data/1015/CSMax_ibias.csv", csv_ascii);
		vec b0V;
		b0V.load("../data/1015/CSMax_globalbias.csv", csv_ascii);
		_CS_b0S = b0V[0];

		/// print model paramters
		cout << "# of users:" << _CS_XM.n_cols << endl;
		cout << "# of items:" << _CS_XM.n_cols << endl;
		cout << "# of dimension:" << _CS_XM.n_rows << endl;

	}

	/*
	* model by SSMax application
	* the model is to generate personalized utility given user-item pair
	*
	*/
	void loadSSModel() {
		cout << ">>> load social surplus maximization model by SSMax" << endl;
		mat tmpMat;
		tmpMat.load(_ssModelFilePrefix + "_umat.csv", csv_ascii);
		_SS_XM = tmpMat.t();
		tmpMat.load(_ssModelFilePrefix + "_imat.csv", csv_ascii);
		_SS_YM = tmpMat.t();
		_SS_buV.load(_ssModelFilePrefix + "_ubias.csv", csv_ascii);
		_SS_biV.load(_ssModelFilePrefix + "_ibias.csv", csv_ascii);
		vec b0V;
		b0V.load(_ssModelFilePrefix + "_globalbias.csv", csv_ascii);
		_SS_b0S = b0V[0];
		/// print model paramters
		cout << "# of SS users:" << _SS_XM.n_cols << endl;
		cout << "# of SS items:" << _SS_YM.n_cols << endl;
		cout << "# of SS dimension:" << _SS_XM.n_rows << endl;
	}

};


int main(int argc, char** argv) {
	if (argc != 5) {
		cout << "usage: SSEval <train_file> <SS model file prefix> <rec_list_file> <cost_percent>" << endl;
	}
	string trainFile = argv[1];
	string ssModelFilePrefix = argv[2];
	string recListFile = argv[3];
	double costPercent = stod(argv[4]);

	SSEval evaluator(trainFile,ssModelFilePrefix,recListFile,costPercent);
	evaluator.sumRecSS();
	cout << "total recommendation social surplus ($): " << evaluator._totalSS << endl;
}