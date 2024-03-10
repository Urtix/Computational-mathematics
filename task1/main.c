#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

int main()
{
    int threads = 4;
    double t1 = omp_get_wtime();
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

    srand(time(NULL));

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
    double dmax; // максимальное изменение значений u
    int k = 0;

    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);
    omp_set_num_threads(threads);
    do
    {
        k++;
        dmax = 0.0;
        int i, j;
        double temp;
        double *dm = calloc(N, sizeof(*dm));
        double d;
        //
        for (int nx = 1; nx < N + 1; nx++)
        {
            dm[nx] = 0;
#pragma omp parallel for shared(u, nx, dm) private(i, j, temp, d)
            for (i = 1; i < nx + 1; i++)
            {
                j = nx - i;
                temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                d = fabs(temp - u[i][j]);
                if (dm[i] < d)
                    dm[i] = d;
            } // конец параллельной области
        }
        // затухание волны
        for (int nx = N - 2; nx > -1; nx--)
        {
#pragma omp parallel for shared(u, nx, dm) private(i, j, temp, d)
            for (i = N - nx + 1; i < N + 1; i++)
            {
                j = 2 * N - nx - i;
                temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                d = fabs(temp - u[i][j]);
                if (dm[i] < d)
                    dm[i] = d;
            } // конец параллельной области
        }
#pragma omp parallel for shared(N, dm, dmax) private(i)
        for (i = 1; i < N + 1; i++)
        {
            omp_set_lock(&dmax_lock);
            if (dmax < dm[i])
                dmax = dm[i];
            omp_unset_lock(&dmax_lock);
        }
    } while (dmax > eps);
    double t2 = omp_get_wtime();
    double time = t2 - t1;

    printf("N: %d\n", N);
    printf("thread: %d\n", threads);
    printf("time: %f sec \n", time);
    printf("iteration: %d\n", k);

    return 1;
}
