#include <iostream>
#include "../src/dataset.cpp"

using namespace std;


int main() {
    Dataset dataset("../data/");
    dataset.loadData();

    dataset.shuffle();
    printf("Done shuffling:\n");
    Batch batch = dataset.getBatch(5, 0);
    for (int i = 0; i < 5; i++) {
        cout << "Label " << i << ": " << (int)(batch.labels[i]) << endl;
        cout << "First 10 pixels: ";
        for (int j = 0; j < 10; j++) {
            cout << batch.images[i * 3072 + j] << " ";
        }
        cout << endl;
    }

    return 0;
}