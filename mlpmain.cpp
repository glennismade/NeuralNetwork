// Simple Multi Layer Perceptron program ...  for SE2NN11
//
// Has single/multi layer network with multiple inputs and outputs,
// Can have linear or sigmoidal activation
// Configuration is set by files containing data sets used 
// Can be tested on simple logic problems or numerical problems
// Dr Richard Mitchell 15/12/06 ... 18/09/13 .. 24/09/15
// Adapted by << Enter Your Name >>

#include <iostream>
#include <fstream>
#include <iomanip>
#include "mlplayer.h"
	// define the classes for layers of neurons and for datasets
#include <windows.h>

using namespace std;

double learnRate = 0.2;
double momentum = 0;

HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

void TestTheNet (LinearLayerNetwork *net, dataset &data, int howprint) {
	// pass each item in the data set, data, to network, net, 
	// and store the outputs which were calculated back into data
	// if howprint = 0; then just print SSE / % classifications; 
	// if howprint = 1, then also print ins/outs
	// if howprint = -1, print nothing
	net -> ComputeNetwork (data);
				// pass whole of dataset to network	
	if (howprint >= 0) data.printdata(howprint);
}				// if appropriate print result suitably


void SetTheWeights (LinearLayerNetwork *net, int nopt) {
	// if called this initialises network net with specific weights
	// some weights array are defined for xor and three logic probs
	vector<double> pictonweights = {0.862518, -0.155797,  0.282885,
							  0.834986, -0.505997, -0.864449,
							  0.036498, -0.430437,  0.481210};
								// weights given in Picton's book for XOR problem only
	vector<double> logweights = {0.2, 0.5, 0.3, 0.3, 0.5, 0.1, 0.4, 0.1, 0.2};
						// initial weights for neuron layer as used in lectures
	vector<double> nlsweights = {0.2, 0.5, 0.4, 0.1, 0.2, 0.4,
							-0.1, 0.5, 0.7, -0.2, -0.7, 0, 0, 0.1, -0.3, 0.3, 0.4, 0.1,
							0.1, 0.2, 0.3, 0.2, 0.1, 0.2,-0.2, 0.4, -0.5, 0.1};
	                    // initial weighst for non lin sep
	if (nopt == 'X')							// if doing XOR problem init with pictonweights
		net -> SetTheWeights (pictonweights);
	else if (nopt == 'O')						// for non linear sep problem
		net -> SetTheWeights (nlsweights);
	else if ( (nopt == 'L') || (nopt == 'S') )	// if doing logic problems, init with logweights
		net -> SetTheWeights (logweights);
						// otherwise the default (random) weights are left unchanged
}

LinearLayerNetwork * MakeNet (char nopt, int numhids, dataset &data) {
		// create and return appropriate network type
		// nopt specifies linear activation/sigmoidal or multi layer
		// number of inputs/outputs are defined in data set data
		// for multilayer, numhids has number of nodes in hidden layers
	if (nopt == 'L')				// if specify linear layer
		return new LinearLayerNetwork (data.numIns(), data.numOuts());
								// call constructor for LinearLayerNetwork
	else if (nopt == 'S')			// if specify sigmoidal layer
		return new SigmoidalLayerNetwork (data.numIns(), data.numOuts());
								// call constructor for SigmoidalLayerNetwork
	else						// if multi-layer
		return new MultiLayerNetwork (data.numIns(), numhids, 
		                   new SigmoidalLayerNetwork (numhids, data.numOuts()) );
								// call constructor for SigmoidalLayerNetwork
}								// whose next layer is a SigmoidalLayerNetwork

char getcapch(void) {
	// gets next character input by user from console, 
	// if lower case convert to equiv upper case
	char och;
	cin >> och;								// get character
	cin.ignore(1);							// skip rest of line

	if ( (och >= 'a')  && (och <= 'z')) och = och - 32;
											// if lower case convert
	return och;
}

void showweights (LinearLayerNetwork *net) {
	// function to print the weights of the neurons in the network net
			// first get space for sufficient number of weights
	vector<double> warray(net->HowManyWeights(), 0);
			// next get the weights from the network
	warray = net->ReturnTheWeights ();
			// now print them out in turn
	for (int ct=0; ct<net->HowManyWeights(); ct++)
		cout << warray[ct] << ',';
	cout << '\n';
}

void setlparas () {					// allow user to enter learning rate and momentum
	cout << "Enter l-rate & momentum (both in range 0..1) > ";
	cin >> learnRate >> momentum;
	cin.ignore(1);
}

void testnet (char nopt, int wopt, int nhid, char *filename, char *dataname) {
	// routine to test the network
	// nopt is network option selected
	// wopt is 0 if specific initial weights are to be used, otherwise default random ones
	// s is name of file with training data

	int emax, esofar = 0;		// maximum number of epochs at a time, and total number of epochs so far
	char och = ' ';				// character used for commands input by user

	dataset data (filename, dataname);				// get data sets from file

	if (data.numIns() == 0)  {
		cout << dataname << " file not found : may be in wrong folder\n";
		return;					// abort function
	}

	srand(wopt);					// initialise random number generator
	if (nopt == 'L') emax = 7; else emax = 1001;
									// if sigmoidal/XOR have 1001 epochs .. else 7

	LinearLayerNetwork *net = MakeNet (nopt, nhid, data);
									// create appropriate form of network
	if (wopt == 0) SetTheWeights (net, nopt);
			// if not random weights, initialise weights as appropriate.

	TestTheNet (net, data, 1);		// test untrained net and print results unless Classifier

	while (och != 'A') {			// loop as follows depending on user commands
		cout << "Enter (L)earn, (P)resent data, (C)hange learn consts, find (W)eights, (S)ave learntdata, (A)bort >";
		och = getcapch();

		if (och == 'L') {			// choose to learn, so pass to network for emax epochs
			for (int ct = 0; ct < emax; ct++) {
				net->AdaptNetwork(data, learnRate, momentum);
				// pass data to network and update weights; print if needed

				if ((emax < 10) || (ct % 200 == 0)) {
							// print SSE every 200'th epoch if sigmoidal, else each time
					cout << "   Epoch " << setw(6) << ct + esofar << " ";
					data.printdata(0);
				}
			}
			esofar = esofar + emax - 1;		// update how many epochs taught so far
		}
		else if (och == 'C')
			setlparas();
		else if (och == 'P') 		// choose to pass training data to network and print
			TestTheNet (net, data, 1); 	
		else if (och == 'W')		// choose to display weights of neurons
			showweights (net);
		else if (och == 'S') {		// choose to save data file 
			TestTheNet (net, data, -1);	// pass to network
			data.savedata();			// save to file
		}
	}
}

void classtest (int numhid, int emax, int wopt, char *tstr, char *ustr) {
	// test network on the classification problem
	// specified are the learning rate, momentum, number of hidden neurpons
	// emax is max number of epochs for learning
	// wopt is seed used to initialise random number generators used to initialise weights
	// data files names are in training and unseen sets are in tstr and ustr
	double vwas = 1000, vsum = 0;			// previous sum of valid.SSE, current sum
	srand(wopt);							// initialise random number generator
    dataset train (tstr, "IrisTrain");		// create training set
	dataset unseen (ustr, "IrisUnseen");	// and unseen set
	LinearLayerNetwork * net = MakeNet ('N', numhid, train);
											// create network with given hidden neurons
	TestTheNet (net, train, 0);				// test training set on untrained net
	TestTheNet (net, unseen, 0);			// and unseen set
	int epoch = 0, learnt = 0;
	while ( (learnt == 0) && (epoch < emax) ) {			// keep going til stop training
		net -> AdaptNetwork (train, learnRate, momentum);		// pass training set
		if (epoch % 50 == 0) {							// every 50th epoch
			cout << " " << epoch;						// print epoch
			train.printarray (" ", 'C', 0, -1);			// and the number of correct classifications
		}
		epoch++;										// increase epoch count
	}
	TestTheNet (net, train, 0);				// test training set on trained net : 0 means just print % classifications
	train.savedata(1);						// save data in data file, so can do tadpole plot
	TestTheNet (net, unseen, 0);			// and unseen set
	unseen.savedata(1);						
}

void numtest (int numhid, int emax, int usevalid, int wopt, char *tstr, char *vstr, char *ustr) {
	// test network on the numerical problem
	
	srand(wopt); //calls the random number generator function (allready within the network)
	dataset numTrain (tstr, "TestTrain"); //opens the dataset file iristrain and passes it to the network
	dataset numUnseen(ustr, "TestUnseen"); //opens the dataset file irisunseen and passed it to the network
	dataset validateSSE(vstr, "Validate");
	double ssePrev = 1000, sseCurrent = 0;
	LinearLayerNetwork * net = MakeNet('N', numhid, numTrain); //calls the createnetwork function and assignes it to the Makenet method/ 

	TestTheNet(net, numTrain, 0); 
	TestTheNet(net, numUnseen, 0);
	TestTheNet(net, validateSSE, 0);
	//presents the datasets to the network for training

	int epoch = 0, learnt = 0; //sets the learning rate and number of epochs to 0 prior to running

	while ((learnt == 0) && (epoch < emax)) {				 // tells the network to loop through the learning function for a given number or epochs
		net->AdaptNetwork(numTrain, learnRate, momentum);	// passes the training set to the network
		if (usevalid) {										// checks is usevalid is true, calls TestTheNet and compares the current SSE to the validate dataset and points to the total SSE.
			TestTheNet(net, validateSSE, -1);
			sseCurrent += validateSSE.TotalSSE();

			if (epoch > 150 && epoch % 10 == 0) // checks to see if the epochs are greater than 100, and if the epochs divided by 10 are = to 0 and if they are checks if the current SSE is greater than the previouse SSE when its multiplied by 0.999
			{
				if (sseCurrent > ssePrev * 0.999) { // if current SSE is equal to previouse SSE x by 0.999 it sets the learnt veriable to true / 1
					learnt = 1;
									
				}
				ssePrev = sseCurrent; // else it sets the previouse sse to be equal to current SSE and sets the current sse to 0 to contnue the loop
				sseCurrent = 0;

				}

		}

		if (epoch % 20 == 0) {							//checks to see if the conditions are met (epochs are set to check every 20)
			cout << " " << epoch;						// returns the number of epochs taken so far (for given number of epochs)
			numTrain.printarray(" ", 'C', 0, -1);			// calls the instance of the dataset and points it to the printarray function 
		}
		epoch++;										// increase epoch count by 1 every time it loops
	}

			cout << " " << epoch;						// adds an aditional print of the epochs and SSE to the screen incase it terminates at an odd number of epochs 
			numTrain.printarray(" ", 'C', 0, -1);

	TestTheNet(net, numTrain, 0);				 // prints the dataset results to the network and returns the SSE
	numTrain.savedata(1);						// saves the results of the dataset to the file irisTrainfull
	TestTheNet(net, numUnseen, 0);			   // does the same as the above but for the unseen set
	numUnseen.savedata(1);					  // does the same as the above but for the unseen set
	TestTheNet(net, validateSSE, 0);		 // prints the results of the validation function and returns the SSE
	validateSSE.savedata(1);
}

void main() {
					// function to create and test single/multi layer network
	int wopt = 0, hopt = 10, emax = 1001;		// initialse key options
	char och = 'A', nopt = 'L', usevalid = 'Y';		// characters used for selecting options

	SMALL_RECT windowSize = {0, 0, 180, 180};
	SetConsoleWindowInfo(hConsole, TRUE, &windowSize);

	cout << "RJMs Perceptron network adapted by RJM\n";

	while (och != 'Q') {						// loop til quit
		cout << "\nSelected Network is ";				// display current set up
		if (nopt == 'L') cout << "Linear Layer";
		else if (nopt == 'S') cout << "Sigmoidal Layer";
		else if (nopt == 'X') cout << "for XOR";
		else if (nopt == 'O') cout << "for other NonLinSep problem";
		else cout << "for Lin Prob with " << hopt << " hidden neurons";
		cout << ". \nInitial random weights seed is " << wopt << "; ";
		cout << "Learning rate is " << learnRate << " and Momentum " << momentum << "\n";

		cout << "\nSelect (T)est network, set (N)etwork, set learning (C)onstants, (I)nitialise random seed, or (Q)uit > ";
												// specify user's options
		och = getcapch();						// read user's choice
		if (och == 'N') {						// user to choose network
			cout << "\nSelect (L)inear layer (S)igmoidal (X)OR (O)ther nonseparable (C)lassifier (N)umerical Problem > ";
			nopt = getcapch();						// get network type
			if ( (nopt != 'L') && (nopt != 'S') && (nopt != 'X') && (nopt != 'O') && (nopt != 'U')) {
			    cout << "Enter number of nodes in hidden layer > ";
				cin >> hopt;		// if approp, get num hidden nodes also
				cin.ignore(1);
			    cout << "Enter max number of epochs for learning >";
				cin >> emax;		
				cin.ignore(1);
				if  (nopt != 'C') {   // for numerical problem, ask about using validation
				  cout << "Use validation set to stop learning (Y/N) > ";
				  usevalid = getcapch();
				} 
			}
		}
		else if (och == 'C') setlparas();
		else if (och == 'I') {					// user to specify initial weights
			if ( (nopt != 'L') && (nopt != 'S') && (nopt != 'X') && (nopt != 'O')) 
				cout << "\nEnter seed used for random weights> ";
			else
				cout << "\nEnter 0 to use weights in notes, else set random weights> ";
   			cin >> wopt;
			cin.ignore(1);
		}
		else if (och == 'T') {					// option to test network
			if ( (nopt == 'L') || (nopt == 'S') )	// test single layer network
				testnet (nopt, wopt, 4, "logdata.txt", "AndOrXor");
			else if (nopt == 'X')				// test MLP on XOR
				testnet (nopt, wopt, 2, "xordata.txt", "XOR");
			else if (nopt == 'O')				// test MLP on XOR
				testnet (nopt, wopt, 4, "nonlinsep.txt", "NonLinSep");
			else if (nopt == 'C')				// test MLP on XOR
				classtest (hopt, emax, wopt, "iristrain.txt", "irisunseen.txt");
			else if (nopt == 'U')				// test MLP on XOR
				testnet (nopt, wopt, 4, "username.txt", "Username");
			else if(nopt == 'M')								// test on numerical problem normalised
				numtest (hopt, emax, usevalid == 'Y', wopt, "trainNorm.txt", "validNorm.txt", "unseenNorm.txt");
			else								// test on numerical problem
				numtest (hopt, emax, usevalid == 'Y', wopt, "GlennTrain.txt", "GlennValid.txt", "GlennUnseen.txt");
		}
	}

	
}
