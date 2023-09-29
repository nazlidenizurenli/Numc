#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(_GNUC) && (defined(x86_64) || defined(i386_))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * _m256d _mm256_add_pd (_m256d a, __m256d b)
 * _m256d _mm256_sub_pd (_m256d a, __m256d b)
 * _m256d _mm256_fmadd_pd (_m256d a, __m256d b, __m256d c)
 * _m256d _mm256_mul_pd (_m256d a, __m256d b)
 * _m256d _mm256_cmp_pd (_m256d a, __m256d b, const int imm8)
 * _m256d _mm256_and_pd (_m256d a, __m256d b)
 * _m256d _mm256_max_pd (_m256d a, __m256d b)
*/
/* Generates a random double between low and high */
double rand_double(double low, double high)
{
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int i = (row * mat->cols) + col;
    return mat->data[i];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO

    int i = (row * mat->cols) + col;
    mat->data[i] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0) 
    {
       return -1; 
    }
    matrix *newmat = (matrix *)malloc(sizeof(matrix));
    if (newmat == NULL)
    {
        return -2;
    }
    newmat->data = (double *)calloc((rows * cols), sizeof(double));
    if (newmat->data == NULL)
    {
        return -2;
    }
    newmat->rows = rows;
    newmat->cols = cols;
    newmat->parent = NULL;
    newmat->ref_cnt = 1;
    *mat = newmat;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL)
    {
        mat->ref_cnt--;

        if (mat->ref_cnt == 0)
        {
            if (mat->data !=NULL){
                free(mat->data);
            }
            if (mat!=NULL){
                free(mat);
            }
        }
    }
    else
    {
        deallocate_matrix(mat->parent);
        mat->parent = NULL;
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows <= 0 || cols <= 0)
    {
        return -1;
    }
    matrix *newmat = (matrix *)malloc(sizeof(matrix));
    if (newmat == NULL)
    {
        return -2;
    }
    if ((rows > from->rows) || (cols > from->cols))
    {
        return -1;
    }
    newmat->data = &from->data[offset];
    newmat->rows = rows;
    newmat->cols = cols;
    newmat->parent = from;

    from->ref_cnt++;

    newmat->ref_cnt = 1;

    *mat = newmat;

    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int mat_size = mat->rows * mat->cols;
    for (int i=0; i < mat_size; i++) {
        mat->data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */

int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    if ((result->rows != mat->rows) || (result->cols != mat->cols))
    {
        return -1;
    }
    int mat_size = mat->rows * mat->cols;
    for (int i = 0; i < mat_size; i++)
    {
        result->data[i] = fabs(mat->data[i]);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    
    int mat_size = mat1->rows * mat1->cols;
    result->rows = mat1->rows;
    result->cols = mat1->cols;
    for (int i = 0; i < mat_size; i++)
    {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    // Task 1.6 TODO

    if (mat1->cols != mat2->rows)
        return -1;

    int c1 = mat1->cols;
    int r1 = mat1->rows;
    int c2 = mat2->cols;
    int r2 = mat2->rows;

    int i = 0;
    int j = 0;
    int k = 0;
    double cij = 0;

    for (i = 0; i < r1; i++)
    {
        for (j = 0; j < c2; j++)
        {
            cij = 0;
            for (k = 0; k < r2; k++)
            {
                cij += mat1->data[(i * c1) + k] * mat2->data[j + (k * c2)];
            }
            result->data[(i * c2) + j] = cij;
        }
    }

    result->rows = mat1->rows;
    result->cols = mat2->cols;
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */

int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int i = 0;
    if (pow == 0) {
	    for(i = 0; i < mat->rows; i++) {
		    for (int j = 0; j <mat->cols; j++) {
			    if(i == j) {
				    result->data[(i*mat->cols)+j]= 1;
			    }
			    else {
				    result->data[(i*mat->cols)+j] = 0;
			    }
		    }
	    }
	    return 0;
    }
    if (pow == 1) {
	    for(i = 0; i < mat->cols * mat->rows; i++){
		   result->data[i] = mat->data[i];
	    }
	    result->cols = mat->cols;
	    result->rows = mat->rows;
	    return 0;
    }
    for (i = 0; i < pow - 1; i++) {
        if (i == 0) {
            mul_matrix(result, mat, mat);
        }
        else{
		matrix *temp;
		allocate_matrix(&temp, result->rows, result->cols);
		for(int n = 0; n < result->cols * result->rows; n++){
			temp->data[n] = result->data[n];
		}
            mul_matrix(result, mat, temp);
        }
    }
    return 0;
}