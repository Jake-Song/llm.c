#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define M_PI 3.14159265358979323846 // redefine the value of PI so that it is equivalent to math.pi in Python
// reference
// def apply_scaling(freqs: torch.Tensor):
//     # Values obtained from grid search
//     scale_factor = 8
//     low_freq_factor = 1
//     high_freq_factor = 4
//     old_context_len = 8192  # original llama3 length

//     low_freq_wavelen = old_context_len / low_freq_factor
//     high_freq_wavelen = old_context_len / high_freq_factor
//     new_freqs = []
//     for freq in freqs:
//         wavelen = 2 * math.pi / freq
//         if wavelen < high_freq_wavelen:
//             new_freqs.append(freq)
//         elif wavelen > low_freq_wavelen:
//             new_freqs.append(freq / scale_factor)
//         else:
//             assert low_freq_wavelen != high_freq_wavelen
//             smooth = (old_context_len / wavelen - low_freq_factor) / (
//                 high_freq_factor - low_freq_factor
//             )
//             new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
//     return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

void apply_scaling(float* freqs) {
    // Values obtained from grid search
    float scale_factor = 8;
    float low_freq_factor = 1;
    float high_freq_factor = 4;
    float old_context_len = 8192;  // original llama3 length

    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;
    for (int i = 0; i < sizeof(freqs); i++) {
        float freq = freqs[i];
        float wavelen = 2 * M_PI / freq;
        if (wavelen < high_freq_wavelen) {
            freqs[i] = freq;
        } else if (wavelen > low_freq_wavelen) {
            freqs[i] = freq / scale_factor;
        } else {
            assert(low_freq_wavelen != high_freq_wavelen);
            float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq;
        }
    }
}

void precompute_freqs_cis(float* freqs_cis, int dim, int end, float theta, int use_scaled) {
    // dim: head size
    // end: sequence length
    // freqs_cis shape: (end, dim)
    float freqs[dim / 2];
    for (int i = 0; i < dim / 2; i++) {
        float j = i;
        freqs[i] = 1.0 / (pow(theta, j*2 / dim));
    }
    if (use_scaled) {
        // TODO: implement apply_scaling
        apply_scaling(freqs);
    }
    for (int i = 0; i < end; i++) {
        float t = i;
        // seek to the input position freqs_cis[i,:]
        float* freqs_cis_i = freqs_cis + i * dim;
        for (int j = 0; j < dim; j+=2) {
            // calculate the cos and sin values
            freqs_cis_i[j] = cos(t * freqs[j/2]); 
            freqs_cis_i[j+1] = sin(t * freqs[j/2]);
        }
    }
}

void apply_rotary_emb(float* xq_inp, float* xk_inp, 
                      float* freqs_cis, 
                      float* xq_out, float* xk_out, 
                      int B, int T, int C, int NH) {
    // NH: number of heads
    // HS: head size
    // inp shape: (B, T, NH, HS)
    // freqs_cis shape: (T, HS)
   
    int hs = C / NH; // head size
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position freqs_cis[t,:]
            float* freqs_cis_bt = freqs_cis + t * hs;

            for (int h = 0; h < NH; h++) {
                // seek to the input position inp[b,t,:]
                float* q = xq_inp + b * T * C + t * C + h * hs; 
                float* k = xk_inp + b * T * C + t * C + h * hs;
                float* xq_out_bt = xq_out + b * T * C + t * C + h * hs;
                float* xk_out_bt = xk_out + b * T * C + t * C + h * hs;
                
                for (int i = 0; i < hs; i+=2) {
                    // apply the rotation
                    // v2i = cos(θ) * v2i - sin(θ) * v2i+1
                    // v2i+1 = sin(θ) * v2i + cos(θ) * v2i+1
                    xq_out_bt[i] = freqs_cis_bt[i] * q[i] - freqs_cis_bt[i+1] * q[i+1];
                    xq_out_bt[i+1] = freqs_cis_bt[i+1] * q[i] + freqs_cis_bt[i] * q[i+1];
                    xk_out_bt[i] = freqs_cis_bt[i] * k[i] - freqs_cis_bt[i+1] * k[i+1];
                    xk_out_bt[i+1] = freqs_cis_bt[i+1] * k[i] + freqs_cis_bt[i] * k[i+1];
                }
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
    // llama3 configuration
    // n_embd = 4096
    // n_head = 32
    // block_size = 8192
    int B = 4; // batch
    int T = 8192; // time / sequence length
    int NH = 32; // number of heads
    int HS = 8; // head size
    int C = NH * HS; 
    float theta = 500000.0;

    float* x = (float*) malloc(B * T * NH * HS * sizeof(float));
    float* q = (float*) malloc(B * T * NH * HS * sizeof(float));
    float* k = (float*) malloc(B * T * NH * HS * sizeof(float));
    float* freqs_cis_real = (float*) malloc(T*HS * sizeof(float));
    
    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }
    fread(freqs_cis_real, sizeof(float), T*HS, file);
    fread(x, sizeof(float), B * T * NH * HS, file);
    fread(q, sizeof(float), B * T * NH * HS, file);
    fread(k, sizeof(float), B * T * NH * HS, file);
    
    fclose(file);

    float* c_freqs_cis = (float*) malloc(T*HS * sizeof(float));
    float* c_q_out = (float*) malloc(B * T * NH * HS * sizeof(float));
    float* c_k_out = (float*) malloc(B * T * NH * HS * sizeof(float));

    precompute_freqs_cis(c_freqs_cis, HS, T, theta, 1);
    apply_rotary_emb(x, x, c_freqs_cis, c_q_out, c_k_out, B, T, C, NH);

    // check correctness of precompute_freqs_cis
    check_tensor(freqs_cis_real, c_freqs_cis, T*HS, "freqs_cis");
    
    // check correctness of apply_rotary_emb
    check_tensor(q, c_q_out, B*T*NH*HS, "q_out");
    check_tensor(k, c_k_out, B*T*NH*HS, "k_out");
       
    free(x);
    free(q);
    free(k);
    free(freqs_cis_real);
    
    return 0;
}

