#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main() {
    int N = 100;
    double f[N + 2][N + 2];
    double h = 1.0 / (N + 1);
    double default_f = 0;
    printf("hhhhhhhhhhhhh  %f     \n",h);
    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            double x = (double)i * h;
            double y =(double) j * h;
            if (x == 0.0) {
                f[i][j] = 100 - 200 * y;
            } else if ( x == 1.0) {
                f[i][j] = -100 + 200 * y;
            } else if ( y == 0.0) {
                f[i][j] = 100 - 200 * x;
            } else if ( y == 1.0) {
                f[i][j] = -100 + 200 * x;
            } else {
                f[i][j] = default_f;
            }
        }
    }

    srand(time(NULL));
    double u[N + 2][N + 2];

    for (int i = 0; i < N+2; i++)
    {
        for (int j = 0; j < N+2; j++)
        {
            u[i][j] = (double)(rand() % (100 - (-100) + 1) + (-100));
        }
    }
    double eps = 0.1;
    double dmax ; // максимальное изменение значений u
    int k = 0;
    do {
        k ++;
        dmax = 0.0;
        for ( int i=1; i < N + 1; i++ )
            for ( int j=1; j < N + 1; j++ ) {
                double temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j]+u[i + 1][j]+ u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
                double dm = fabs(temp-u[i][j]);
                if ( dmax < dm ) dmax = dm;
            }
        if (k == 100) break;
    } while ( dmax > eps );

    printf("%d",k);
    return 1;
}
