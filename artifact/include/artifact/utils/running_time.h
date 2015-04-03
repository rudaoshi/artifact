#ifndef ARTIFACT_UTILS_RUNNING_TIME_H
#define ARTIFACT_UTILS_RUNNING_TIME_H

#include <time.h>


#ifndef speedtest__
#define speedtest__(data)   for (long blockTime = NULL; (blockTime == NULL ? (blockTime = clock()) != NULL : false); printf(data "%.9fs", (double) (clock() - blockTime) / CLOCKS_PER_SEC))
#endif

#endif