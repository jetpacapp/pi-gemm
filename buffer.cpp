//
//  bbuffer.cpp
//  jpcnn
//
//  Created by Peter Warden on 1/9/14.
//  Copyright (c) 2014 Jetpac, Inc. All rights reserved.
//

#include "buffer.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include "mailbox.h"


Buffer::Buffer(const Dimensions& dims) :
  _dims(dims),
  _name(NULL),
  _debugString(NULL),
  _bitsPerElement(32)
{
  const int elementCount = _dims.elementCount();
  const size_t byteCount = (elementCount * sizeof(float));
  _gpuMemoryHandle = mem_alloc(byteCount, 4096, GPU_MEM_FLG);
  if (!_gpuMemoryHandle) {
    fprintf(stderr, "Unable to allocate %d bytes of GPU memory\n", byteCount);
  }
  _gpuMemoryBase = mem_lock(_gpuMemoryHandle);
  _data = (float*)(mapmem(_gpuMemoryBase + GPU_MEM_MAP, byteCount));
  _doesOwnData = true;
  setName("None");
}

Buffer::~Buffer()
{
  if (_doesOwnData) {
    if (_data) {
      const int elementCount = _dims.elementCount();
      const size_t byteCount = (elementCount * sizeof(float));
      unmapmem(_data, byteCount);
    }
    mem_unlock(_gpuMemoryHandle);
    mem_free(_gpuMemoryHandle);
  }
  if (_debugString)
  {
    free(_debugString);
  }
  if (_name)
  {
    free(_name);
  }
}

char* Buffer::debugString() {
  if (!_debugString) {
    _debugString = (char*)(malloc(MAX_DEBUG_STRING_LEN));
  }
  snprintf(_debugString, MAX_DEBUG_STRING_LEN, "Buffer %s - %s", _name, _dims.debugString());
  return _debugString;
}

void Buffer::printContents(int maxElements) {
  FILE* output = stderr;
  fprintf(output, "%s : \n", debugString());
  Dimensions dims = _dims;
  const int elementCount = dims.elementCount();
  if (elementCount < 5000) {
    maxElements = -1;
  }
  bool isSingleImage = (dims[0] == 1);
  if (isSingleImage) {
    dims = dims.removeDimensions(1);
  }
  const int dimsLength = dims._length;
  const float* const data = _data;
  if (isSingleImage) {
    fprintf(output, "[");
  }
  if (dimsLength == 1) {
    const int width = dims[0];
    int xLeft;
    int xRight;
    if (maxElements < 1) {
      xLeft = -1;
      xRight = -1;
    } else {
      xLeft = (maxElements / 2);
      xRight = (width - (maxElements / 2));
    }
    fprintf(output, "[");
    for (int x = 0; x < width; x += 1) {
      if (x == xLeft) {
        fprintf(output, "...");
      }
      if ((x >= xLeft) && (x < xRight)) {
        continue;
      }
      if (x > 0) {
        fprintf(output, ", ");
      }
      const float value = *(data + dims.offset(x));
      fprintf(output, "%.10f", value);
    }
    fprintf(output, "]");
  } else if (dimsLength == 2) {
    const int height = dims[0];
    const int width = dims[1];

    int yTop;
    int yBottom;
    if (maxElements < 1) {
      yTop = -1;
      yBottom = -1;
    } else {
      yTop = (maxElements / 2);
      yBottom = (height - (maxElements / 2));
    }

    int xLeft;
    int xRight;
    if (maxElements < 1) {
      xLeft = -1;
      xRight = -1;
    } else {
      xLeft = (maxElements / 2);
      xRight = (width - (maxElements / 2));
    }

    fprintf(output, "[");
    for (int y = 0; y < height; y += 1) {
      if (y == yTop) {
        fprintf(output, "...\n");
      }
      if ((y >= yTop) && (y < yBottom)) {
        continue;
      }
      fprintf(output, "[");
      for (int x = 0; x < width; x += 1) {
        if (x == xLeft) {
          fprintf(output, "...");
        }
        if ((x >= xLeft) && (x < xRight)) {
          continue;
        }
        if (x > 0) {
          fprintf(output, ", ");
        }
        const float value = *(data + dims.offset(y, x));
        fprintf(output, "%.10f", value);
      }
      if (y < (height - 1)) {
        fprintf(output, "],\n");
      } else {
        fprintf(output, "]");
      }
    }
    fprintf(output, "]");
  } else if (dimsLength == 3) {
    const int height = dims[0];
    const int width = dims[1];
    const int channels = dims[2];

    int yTop;
    int yBottom;
    if (maxElements < 1) {
      yTop = -1;
      yBottom = -1;
    } else {
      yTop = (maxElements / 2);
      yBottom = (height - (maxElements / 2));
    }

    int xLeft;
    int xRight;
    if (maxElements < 1) {
      xLeft = -1;
      xRight = -1;
    } else {
      xLeft = (maxElements / 2);
      xRight = (width - (maxElements / 2));
    }

    fprintf(output, "[");
    for (int y = 0; y < height; y += 1) {
      if (y == yTop) {
        fprintf(output, "...\n");
      }
      if ((y >= yTop) && (y < yBottom)) {
        continue;
      }
      fprintf(output, "[");
      for (int x = 0; x < width; x += 1) {
        if (x == xLeft) {
          fprintf(output, "...");
        }
        if ((x >= xLeft) && (x < xRight)) {
          continue;
        }
        if (x > 0) {
          fprintf(output, ", ");
        }
        fprintf(output, "(");
        for (int channel = 0; channel < channels; channel += 1) {
          const float value = *(data + dims.offset(y, x, channel));
          if (channel > 0) {
            fprintf(output, ", ");
          }
          fprintf(output, "%.10f", value);
        }
        fprintf(output, ")");
      }
      if (y < (height - 1)) {
        fprintf(output, "],\n");
      } else {
        fprintf(output, "]");
      }
    }
    fprintf(output, "]");
  } else {
    fprintf(output, "Printing of buffers with %d dimensions is not supported\n", dimsLength);
  }
  if (isSingleImage) {
    fprintf(output, "]");
  }
  fprintf(output, "\n");
}

void Buffer::setName(const char* name) {
  if (_name)
  {
    free(_name);
  }
  size_t byteCount = (strlen(name) + 1);
  _name = (char*)(malloc(byteCount));
  strncpy(_name, name, byteCount);
}

void Buffer::populateWithRandomValues(float min, float max) {
  const int elementCount = _dims.elementCount();
  const int bitsPerElement = _bitsPerElement;
  float* dataStart = _data;
  float* dataEnd = (_data + elementCount);
  float* data = dataStart;
  while (data < dataEnd) {
    // rand() is awful, but as long as we're using this for debugging
    // it should be good enough, and avoids dependencies.
    *data = (((max - min) * ((float)rand() / RAND_MAX)) + min);
    data += 1;
  }
}

bool buffer_are_all_close(Buffer* a, Buffer* b, float tolerance) {

  if (a == NULL) {
    fprintf(stderr, "Buffer a is NULL\n");
    return false;
  }

  if (b == NULL) {
    fprintf(stderr, "Buffer b is NULL\n");
    return false;
  }

  if (a->_dims._length != b->_dims._length) {
    fprintf(stderr, "Buffers have different numbers of dimensions - %s vs %s\n", a->debugString(), b->debugString());
    return false;
  }

  for (int index = 0; index < a->_dims._length; index += 1) {
    if (a->_dims._dims[index] != b->_dims._dims[index]) {
      fprintf(stderr, "Buffers are different sizes - %s vs %s\n", a->debugString(), b->debugString());
      return false;
    }
  }

  int differentCount = 0;
  float totalDelta = 0.0f;
  float* aCurrent = a->_data;
  float* aEnd = (a->_data + a->_dims.elementCount());
  float* bCurrent = b->_data;
  while (aCurrent != aEnd) {
    const float aValue = *aCurrent;
    const float bValue = *bCurrent;
    const float delta = (aValue - bValue);
    const float absDelta = fabsf(delta);
    if (absDelta > tolerance) {
      differentCount += 1;
    }
    totalDelta += absDelta;
    aCurrent += 1;
    bCurrent += 1;
  }

  if (differentCount > 0) {
    const float differentPercentage = 100 * (differentCount / (float)(a->_dims.elementCount()));
    const float meanDelta = (totalDelta / a->_dims.elementCount());
    fprintf(stderr, "Buffers contained %f%% different values (%d), mean delta = %f - %s vs %s\n",
      differentPercentage,
      differentCount,
      meanDelta,
      a->debugString(),
      b->debugString());
    return false;
  }

  return true;
}