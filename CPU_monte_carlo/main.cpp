#include "macro.h"
using namespace std;

int M, N, Nonzeros;
long long head, tail, freq;
int Pages;

int* Ap;	// Pointer(Row)
int* Ai;	// Row
int* Aj;	// Column
int* Rc;	// Count of Nonzeros of each row
float* Av;	// Value of each nonzero
float* R;	// Pagerank value
int* csr_row;

void readData() {
	// Open the file:
	FILE* fin;
	fin = fopen("wikipedia-20051105.mtx", "r");
	if (fin == 0) {
		cout << "Fail to open file!" << endl;
		return;
	}
	char buffer[1000];
	while (fgets(buffer, 1000, fin)) {
		if (buffer[0] != '%')
			break;
	}
	string header = buffer, temp;
	stringstream ss;
	ss << header;
	getline(ss, temp, ' ');
	M = stoi(temp);
	getline(ss, temp, ' ');
	N = stoi(temp);
	getline(ss, temp, ' ');
	Nonzeros = stoi(temp);
	Pages = M;

	Ai = new int[Nonzeros];
	Aj = new int[Nonzeros];
	cout << "Start Reading Dataset..." << endl;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	timer(head);
	int l = 0;
	while (fscanf(fin, "%d %d", &Ai[l], &Aj[l]) != EOF) {
		l++;
	}
	timer(tail);

	if (l != Nonzeros)
	{
		cout << "Read Dataset Corrupted!" << endl;
		return;
	}
	cout << "Read Dataset Success! Time: " << total(head, tail) / 1000000.0 << "s." << endl;
}

void init() {
	int Ap_count = Pages + 1;
	Ap = new int[Ap_count];
	Rc = new int[Pages];
	R = new float[Pages];
	Av = new float[Nonzeros];

	memset(Rc, 0, sizeof(int) * Pages);
	memset(Ap, 0, sizeof(int) * (Ap_count));
	memset(R, 0, sizeof(float) * Pages);

	int j = 0;
	for (int i = 0; i < Nonzeros; i++) {
		Rc[Ai[i]]++;
		while (j < Aj[i]) {
			Ap[++j] = i;
		}
	}
	Ap[Ap_count-1] = Nonzeros;

	csr_row = new int[Ap_count];
	memset(csr_row, 0, (Ap_count) * sizeof(int));
	for (int i = 0; i < Nonzeros; i++)
		csr_row[Ai[i]+1]++;
	for (int i = 0; i < Ap_count-1; i++)
		csr_row[i + 1] += csr_row[i];

	float newval = 0.f;
	int count = 1;
	int begin = 0;
	for (int l = 0; l < Nonzeros; l++) {
		if (l > 0 && Aj[l] != Aj[l-1]) {
			newval = 1.0 / count;
			for (int i = begin; i <= l - 1; i++)
				Av[i] = newval;
			count = 1;
			begin = l;
		}
		else {
			count++;
			if (l == Nonzeros - 1) {
				newval = 1.0 / count;
				for (int i = begin; i < Nonzeros; i++)
					Av[i] = newval;
			}
		}
	}
	cout << "Initialization Complete!" << endl;
}

int main() {
	readData();
	init();
	monte_carlo(Pages, Nonzeros, R, Ai, Aj, Ap, Rc, Av);
	power_iter(Pages, Nonzeros, R, Ai, Aj, csr_row, Rc, Av);
	return 0;
}

