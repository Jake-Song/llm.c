// must run `python rmsnorm.py` first to generate the reference data
// then compile for example as `gcc rmsnorm.c -o rmsnorm -lm`
// and then run as `./rmsnorm` to see the output

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void rmsnorm_forward(float* out, float* mean, float* rsqrt,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-6f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;

            // calculate the mean of the values of power 2
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                // calculate the values of power 2
                float x2 = x[i] * x[i];
                m += x2;
            }
            // calculate the mean 
            m = m/C;
            // calculate the rsqrt
            float s = 1.0f / sqrtf(m + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * x[i]); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rsqrt[b * T + t] = s;
        }
    }
}

void rmsnorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rsqrt,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rsqrt_bt = rsqrt[b * T + t];

            // first: reduce operations
            float dnorm_x_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float dnorm_i = weight[i] * dout_bt[i];
              
                dnorm_x_mean += dnorm_i * inp_bt[i];
            }
            dnorm_x_mean = dnorm_x_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = inp_bt[i] * rsqrt_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i * rsqrt_bt; // term 1
                dval -= dnorm_x_mean * pow(rsqrt_bt, 3) * inp_bt[i]; // term 2
                dinp_bt[i] += dval;
            }
        }
    }
}
    
// poor man's tensor checker
int check_tensor(float *a, float *b, int n, char* label) {
    int ok = 1;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } else {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    return ok;
}

int main() {

    int B = 2; // batch
    int T = 3; // time / sequence length
    int C = 4; // number of channels

    float* x = (float*) malloc(B * T * C * sizeof(float));
    float* w = (float*) malloc(C * sizeof(float));
    float* b = (float*) malloc(C * sizeof(float));
    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* mean = (float*) malloc(B * T * sizeof(float));
    float* rsqrt = (float*) malloc(B * T * sizeof(float));
    float* dout = (float*) malloc(B * T * C * sizeof(float));
    float* dx = (float*) malloc(B * T * C * sizeof(float));
    float* dw = (float*) malloc(C * sizeof(float));
    float* db = (float*) malloc(C * sizeof(float));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    fread(x, sizeof(float), B * T * C, file);
    fread(w, sizeof(float), C, file);
    fread(b, sizeof(float), C, file);
    fread(out, sizeof(float), B * T * C, file);
    fread(mean, sizeof(float), B * T, file);
    fread(rsqrt, sizeof(float), B * T, file);
    fread(dout, sizeof(float), B * T * C, file);
    fread(dx, sizeof(float), B * T * C, file);
    fread(dw, sizeof(float), C, file);
    fread(db, sizeof(float), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    float* c_out = (float*) malloc(B * T * C * sizeof(float));
    float* c_mean = (float*) malloc(B * T * sizeof(float));
    float* c_rsqrt = (float*) malloc(B * T * sizeof(float));
    rmsnorm_forward(c_out, c_mean, c_rsqrt, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rsqrt, c_rsqrt, B*T, "rsqrt");

    // backward pass (note calloc inits grads to zero)
    float* c_dx = (float*) calloc(B * T * C, sizeof(float));
    float* c_dw = (float*) calloc(B * T, sizeof(float));
    float* c_db = (float*) calloc(B * T, sizeof(float));
    rmsnorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rsqrt, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    free(x);
    free(w);
    free(b);
    free(out);
    free(mean);
    free(rsqrt);
    free(dout);
    free(dx);
    free(dw);
    free(db);
    return 0;
}