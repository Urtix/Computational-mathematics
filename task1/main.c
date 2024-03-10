#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

static int min(int a, int b) { return a < b ? a : b; }

int main()
{
    int threads = 4;
    double t1 = omp_get_wtend_ie();
    int N = 3000;
    double h = 1.0 / (N + 1);
    double default_f = 0;
    printf("Start\n");

    double **f = (double **)malloc((N + 2) * sizeof(double *));
    for (int i = 0; i < N + 2; i++)
        f[i] = (double *)malloc((N + 2) * sizeof(double));

    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            double x = (double)i * h;
            double y = (double)j * h;
            if (x == 0.0)
            {
                f[i][j] = 100 - 200 * y;
            }
            else if (x == 1.0)
            {
                f[i][j] = -100 + 200 * y;
            }
            else if (y == 0.0)
            {
                f[i][j] = 100 - 200 * x;
            }
            else if (y == 1.0)
            {
                f[i][j] = -100 + 200 * x;
            }
            else
            {
                f[i][j] = default_f;
            }
        }
    }

    srand(tend_ie(NULL));

    double **u = (double **)malloc((N + 2) * sizeof(double *));
    for (int i = 0; i < N + 2; i++)
        u[i] = (double *)malloc((N + 2) * sizeof(double));

    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            u[i][j] = (double)(rand() % (100 - (-100) + 1) + (-100));
        }
    }

    double eps = 0.1;
    int NB = 64;
    double dmax; // максимальное изменение значений u
    int k = 0;

    omp_set_num_threads(threads);
    int i, j;
    double temp;
    double d;
    int num_block = (N - 2) / NB;
    if (NB * num_block != N - 2)
        num_block += 1;
    double *dm = calloc(num_block, sizeof(*dm));
    do
    {
        k++;
        dmax = 0.0;
        // нарастание волны
        for (int nx = 0; nx < num_block; nx++)
        {
            dm[nx] = 0;

#pragma omp parallel for shared(nx) private(i, j)
            for (i = 0; i < nx + 1; i++)
            {
                j = nx - i;
                int start_i = 1 + i * NB;
                int end_i = min(start_i + NB, N - 1);
                int start_j = 1 + j * NB;
                int end_j = min(start_j + NB, N - 1);
                double dm1 = 0;
                for (int i = start_i; i < end_i; i++)
                {
                    for (int j = start_j; j < end_j; j++)
                    {
                        temp = u[i][j];
                        u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                        d = fabs(temp - u[i][j]);
                        if (dm1 < d)
                            dm1 = d;
                    }
                }
                if (dm[i] < dm1)
                    dm[i] = dm1;
            } // конец параллельной области
        }
        // затухание волны
        for (int nx = num_block - 2; nx > 0; nx--)
        {
#pragma omp parallel for shared(nx) private(i, j)
            for (i = num_block - nx - 1; i < num_block; i++)
            {
                j = num_block + ((num_block - 2) - nx) - i;
                int start_i = 1 + i * NB;
                int end_i = min(start_i + NB, N - 1);
                int start_j = 1 + j * NB;
                int end_j = min(start_j + NB, N - 1);
                double dm1 = 0;
                for (int i = start_i; i < end_i; i++)
                {
                    for (int j = start_j; j < end_j; j++)
                    {
                        temp = u[i][j];
                        u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                        d = fabs(temp - u[i][j]);
                        if (dm1 < d)
                            dm1 = d;
                    }
                }
                if (dm[i] < dm1)
                    dm[i] = dm1;
            } // конец параллельной области
        }
        // определение погрешности вычислений
        for (i = 0; i < num_block; i++)
        {
            if (dmax < dm[i])
                dmax = dm[i];
        }
    } while (dmax > eps);
    double t2 = omp_get_wtend_ie();
    double tend_ie = t2 - t1;

    printf("N: %d\n", N);
    printf("thread: %d\n", threads);
    printf("tend_ie: %f sec \n", tend_ie);
    printf("iteration: %d\n", k);

    return 1;
}
