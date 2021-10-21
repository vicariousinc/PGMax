#include <stdio.h>
#include <stdlib.h>
#define NEG_INF -1000

int f_index_r(int index, int hps, int vps){
    return -hps + (index - 1) /(2*vps+1);
}

int f_index_c(int index, int hps, int vps){
    return -vps + (index - 1)%(2*vps+1);
}

int f_rc_index(int r, int c, int hps, int vps){
    return 1 + c + vps + (2*hps+1)*(r+hps);
}

int max(int a, int b){
    if(a >= b)return a;
    return b;
}

int min(int a, int b){
    if(a <= b)return a;
    return b;
}

void _initialize_phi(int M, int r, int hps, int vps, int *len, int **rows, int** cols){

    int i, j;
    int size = M*(2*r+1)*(2*r+1);

    int *data1, *data2;
    data1 = (int *)malloc(sizeof(int) * size);
    data2 = (int *)malloc(sizeof(int) * size);
    int index = 0;

    for(i=1; i<M; i++){
        int r1 = f_index_r(i, hps, vps);
        int c1 = f_index_c(i, hps, vps);

        int r2_min = max(r1-r, -hps);
        int r2_max = min(r1+r, hps);
        int c2_min = max(c1-r, -vps);
        int c2_max = min(c1+r, vps);

        int r2, c2;
        for(r2=r2_min; r2<=r2_max; r2++){
            for(c2=c2_min; c2<=c2_max; c2++){
                j = f_rc_index(r2, c2, hps, vps);
                data1[index] = i;
                data2[index] = j;
                index += 1;
            }
        }
    }

    *rows = data1;
    *cols = data2;
    *len = index;
}

double* compute_new_belief(int* rows, int* cols, int n, const double *belief, int len_belief){
    double * new_belief = (double *)malloc(sizeof(double) * len_belief);
    int i, j, k;
    int first_init = 1;

    new_belief[0] = belief[0];
    for(i=1;i<len_belief;i++){
        if(new_belief[0] < belief[i])
            new_belief[0] = belief[i];
    }
    new_belief[0] += NEG_INF;

    for(k=0;k<n;k++){
        i = rows[k];
        j = cols[k];

        if(k > 0 && rows[k] != rows[k-1]){
            first_init = 1;
        }

        if(first_init == 1){
            new_belief[i] = belief[j];
            first_init = 0;
        } else {
            if(new_belief[i] < belief[j]){
                new_belief[i] = belief[j];
            }
        }
    }

    return new_belief;
}

void get_phi(int r, int c, int *rows, int *cols, int n, int *ret){
    int low = 0, high = n-1;
    int mid;
    while(low <= high){
        mid = low + (high - low)/2;

        if(rows[mid] == r){
            if(mid == 0 || rows[mid-1] < r){
                int k = mid;
                while(k<n && rows[k] == rows[mid]){
                    if(cols[k] == c){
                        *ret = 0;
                        return;
                    }
                    k += 1;
                }
                *ret = NEG_INF;
                return;

            }

            high = mid - 1;

        }else if(rows[mid] > r)
            high = mid - 1;
        else
            low = mid + 1;

    }
    *ret = NEG_INF;
}


void cleanup(int *data1, int *data2){
    free(data1);
    free(data2);
}


int main(){
	printf("Hello world! \n");
	return 0;
}
