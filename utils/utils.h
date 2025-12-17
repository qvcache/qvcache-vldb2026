// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Adapted from: https://github.com/microsoft/DiskANN

#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include <cstdint>
#include <fstream>
#include <cstring>
#include <algorithm>

// ROUND_UP macro from DiskANN
#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#endif  // UTILS_UTILS_H

