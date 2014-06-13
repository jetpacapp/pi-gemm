//
//  buffer.h
//  jpcnn
//
//  Created by Peter Warden on 1/9/14.
//  Copyright (c) 2014 Jetpac, Inc. All rights reserved.
//

#ifndef INCLUDE_BUFFER_H
#define INCLUDE_BUFFER_H

#include <stdint.h>

#include "dimensions.h"

class Buffer
{
public:

  Buffer(const Dimensions& dims);
  Buffer(const Dimensions& dims, float* data);
  Buffer(const Dimensions& dims, void* quantizedData, float min, float max, int bitsPerElement);
  Buffer(const Dimensions& dims, float min, float max, int bitsPerElement);
  virtual ~Buffer();

  Dimensions _dims;
  float* _data;
  uint32_t _gpuMemoryHandle;
  uint32_t _gpuMemoryBase;
  int _bitsPerElement;
  bool _doesOwnData;
  char* _debugString;
  char* _name;

  void populateWithRandomValues(float min, float max);

  char* debugString();
  void printContents(int maxElements=8);
  void setName(const char*);
};

extern bool buffer_are_all_close(Buffer* a, Buffer* b, float tolerance=0.000001);

#endif // INCLUDE_BUFFER_H
