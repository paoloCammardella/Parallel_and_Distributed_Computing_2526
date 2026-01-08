#include <stdio.h>
#include <omp.h>

int main(){
    int somma = 0;
    int i;

    #pragma omp parallel for
    for(i = 0; i < 10; ++i){
        printf("Thread %d sta aggiungendo %d\n", omp_get_thread_num(), i);

        #pragma omp critical
        {
            somma += i;
        }
    }
    printf("La somma finale = %d\n", somma);
    return 0;
}