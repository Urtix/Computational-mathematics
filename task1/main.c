#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NB 64
#define eps 0.1

static int min(int a, int b) { return a < b ? a : b; }

double **allocate2DArray(int N)
{
    double **array = (double **)malloc((N + 2) * sizeof(double *));

    for (int i = 0; i < N + 2; i++)
    {
        array[i] = (double *)malloc((N + 2) * sizeof(double));
    }

    return array;
}

double u1(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 1000·x^3·y - 2000·y^4+500·y^3+x^2·y^3-700·x+250·y
        return 1000.0 * pow(x, 3) * y - 2000.0 * pow(y, 4) + 500.0 * pow(y, 3) + pow(x, 2) * pow(y, 3) - 700 * x + 250 * y;
    }
    else
    {
        return 0;
    }
}

double f1(double x, double y)
{
    // 6000·x·y+2·y^3  +  6·x^2·y-24000·y^2+3000·y
    return 6000 * x * y + 2 * pow(y, 3) + 6 * pow(x, 2) * y - 24000 * pow(y, 2) + 3000 * y;
}

double u2(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 10x^3 + 20y^3
        return 10.0 * pow(x, 3) * y + 20.0 * pow(y, 3);
    }
    else
    {
        return 0;
    }
}

double f2(double x, double y)
{
    // 60x + 120y
    return 60 * x + 120 * y;
}

double u3(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 100x^3 + 200y^3
        return 100.0 * pow(x, 3) + 200.0 * pow(y, 3);
    }
    else
    {
        return 0;
    }
}

double f3(double x, double y)
{
    // 600x + 1200y
    return 600 * x + 1200 * y;
}

double u4(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 1000x^3 + 2000y^3
        return 1000.0 * pow(x, 3) + 2000.0 * pow(y, 3);
    }
    else
    {
        return 0;
    }
}

double f4(double x, double y)
{
    // 6000x + 12000y
    return 6000 * x + 12000 * y;
}

double u5(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 10x^4 + 20y^4
        return 10.0 * pow(x, 4) * y + 20.0 * pow(y, 4);
    }
    else
    {
        return 0;
    }
}

double f5(double x, double y)
{
    // 120x + 240y
    return 120 * pow(x, 2) + 240 * pow(y, 2);
}

double u6(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 10x^5 + 20y^3
        return 10.0 * pow(x, 5) * y + 20.0 * pow(y, 5);
    }
    else
    {
        return 0;
    }
}

double f6(double x, double y)
{
    // 200x^3 + 400y^3
    return 200 * pow(x, 3) + 400 * pow(y, 3);
}

double u7(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        // 10x^50 + 20y^50
        return 10.0 * pow(x, 50) * y + 20.0 * pow(y, 50);
    }
    else
    {
        return 0;
    }
}

double f7(double x, double y)
{
    // 24500x^48 + 49000y^48
    return 24500 * pow(x, 48) + 49000 * pow(y, 48);
}

void initialize_f_u(double **u, double **f, int N, double h)
{
    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            double x = i * h;
            double y = j * h;
            u[i][j] = u6(i, j, N, x, y);
            f[i][j] = f6(x, y);
        }
    }
}

void update_dm(double **u, double **f, double *dm, int N, double h, int j, int i)
{
    int start_i = 1 + i * NB;
    int end_i = min(start_i + NB, N + 1);
    int start_j = 1 + j * NB;
    int end_j = min(start_j + NB, N + 1);
    double dm1 = 0;
    for (int i = start_i; i < end_i; i++)
    {
        for (int j = start_j; j < end_j; j++)
        {
            double temp = u[i][j];
            u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
            double d = fabs(temp - u[i][j]);
            if (dm1 < d)
                dm1 = d;
        }
    }
    if (dm[i] < dm1)
        dm[i] = dm1;
}

int wave_propagation(int num_block, double **u, double **f, double *dm, int N, double h)
{
    int k = 0;
    double dmax;

    do
    {
        k++;
        dmax = 0.0;

        // Нарастание волны
        for (int nx = 0; nx < num_block; nx++)
        {
            int i, j;
            dm[nx] = 0;

#pragma omp parallel for shared(nx) private(i, j)
            for (i = 0; i < nx + 1; i++)
            {
                j = nx - i;
                update_dm(u, f, dm, N, h, j, i);
            }
        }

        // Затухание волны
        for (int nx = num_block - 2; nx > -1; nx--)
        {
            int i, j;
#pragma omp parallel for shared(nx) private(i, j)
            for (i = num_block - nx - 1; i < num_block; i++)
            {
                j = num_block + ((num_block - 2) - nx) - i;
                update_dm(u, f, dm, N, h, j, i);
            }
        }

        // Определение погрешности вычислений
        for (int i = 0; i < num_block; i++)
        {
            if (dmax < dm[i])
                dmax = dm[i];
        }

    } while (dmax > eps);
    return k;
}

void free_memory(double **u, double **f, double *dm, int N)
{
    for (int i = 0; i < N + 2; i++)
    {
        free(u[i]);
        free(f[i]);
    }
    free(u);
    free(f);
    free(dm);
}

int main()
{
    for (int x = 1; x < 11; x++)
    {
        printf("Experiment number %d. \n", x);
        int grids[5] = {100, 300, 500, 1000, 3000};
        int num_grids = sizeof(grids) / sizeof(grids[0]);
        for (int threads = 1; threads <= 4; threads += 3)
        {
            for (int n = 0; n < num_grids; n++)
            {
                double t1 = omp_get_wtime();
                int N = grids[n];
                double h = 1.0 / (N + 1);

                printf("Start\n");

                double **f = allocate2DArray(N);
                double **u = allocate2DArray(N);
                initialize_f_u(u, f, N, h);

                double dmax; // максимальное изменение значений u
                int k = 0;

                omp_set_num_threads(threads);
                int num_block = (N - 2) / NB + 1;
                double *dm = calloc(num_block, sizeof(*dm));
                k = wave_propagation(num_block, u, f, dm, N, h);
                double t2 = omp_get_wtime();
                double tend_ie = t2 - t1;

                printf("N: %d. ", N);
                printf("Thread: %d. ", threads);
                printf("Time: %f sec. ", tend_ie);
                printf("iteration: %d.\n", k);

                free_memory(u, f, dm, N);
            }
        }
        printf("\n");
    }

    return 1;
}
