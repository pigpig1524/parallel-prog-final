#ifndef MODELS_H
#define MODELS_H

struct Image {
    int width;
    int height;
    float * r_channel;
    float * g_channel;
    float * b_channel;
};


struct Sample {
    unsigned char label;
    float * image;
};

struct Split {
    unsigned char * labels;
    float ** images = nullptr;
};

struct Batch {
    int batchSize;
    unsigned char * labels;
    float * images;
};


#endif // MODELS_H