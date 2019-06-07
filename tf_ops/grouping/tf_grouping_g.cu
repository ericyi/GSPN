#include<stdio.h>


// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*3*batch_index;
    xyz2 += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float x2=xyz2[j*3+0];
            float y2=xyz2[j*3+1];
            float z2=xyz2[j*3+2];
            float x1=xyz1[k*3+0];
            float y1=xyz1[k*3+1];
            float z1=xyz1[k*3+2];
    	    float d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}



// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}


// input: points_feat (b,num_point,c), idx (b,num_query,nsample)
// output: out (b,num_query,c), max_idx (b,num_query,c)
__global__ void group_maxpool_gpu(int b, int num_point, int chan, int num_query, int nsample, const float *points, const int *idx, float *out, int *max_idx) {
    int batch_index = blockIdx.x;
    points += num_point*chan*batch_index;
    idx += num_query*nsample*batch_index;
    out += num_query*chan*batch_index;
    max_idx += num_query*chan*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    float temp_feat;
    int one_max_idx;
    float max_feat=-10000.0;


    for (int j=index;j<num_query;j+=stride) {
        for (int l=0;l<chan;++l) {
            max_feat=-10000.0;
            for (int k=0;k<nsample;++k) {
                int ii = idx[j*nsample+k];
                temp_feat = points[ii*chan+l];
                if(temp_feat>max_feat) {max_feat=temp_feat; one_max_idx=ii;}
            }
            out[j*chan+l] = max_feat;
            max_idx[j*chan+l] = one_max_idx;
        } 
    }
}


// input: grad_out (b,num_query,c), max_idx (b,num_query,c),
// output: grad_points (b,num_point,c)
__global__ void group_maxpool_grad_gpu(int b, int num_point, int chan, int num_query, const float *grad_out, const int *max_idx, float *grad_points) {
    int batch_index = blockIdx.x;
    max_idx += num_query*chan*batch_index;
    grad_out += num_query*chan*batch_index;
    grad_points += num_point*chan*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<num_query;j+=stride) {
        for (int l=0;l<chan;++l) {
            int ii = max_idx[j*chan+l];
            atomicAdd(&grad_points[ii*chan+l], grad_out[j*chan+l]);
        }
    }
}






// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<b,256>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,256>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}
void groupMaxpoolLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out, int *max_idx){
    group_maxpool_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out,max_idx);
    //cudaDeviceSynchronize();
}
void groupMaxpoolGradLauncher(int b, int n, int c, int m, const float *grad_out, const int *max_idx, float *grad_points){
    group_maxpool_grad_gpu<<<b,256>>>(b,n,c,m,grad_out,max_idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}




