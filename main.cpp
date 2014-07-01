#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>
#include <assert.h>
#include <cblas.h>

#include "buffer.h"
#include "mailbox.h"

#define NUM_QPUS        (12)
#define MAX_CODE_SIZE   (8192)
#define NUM_MESSAGE_VALUES (2)

enum JPCBLAS_ORDER {
  JPCblasRowMajor=101,
  JPCblasColMajor=102
};
enum JPCBLAS_TRANSPOSE {
  JPCblasNoTrans=111,
  JPCblasTrans=112,
  JPCblasConjTrans=113,
  JPAtlasConj=114
};

extern uint32_t g_gemm_8bitCode[];
extern size_t g_gemm_8bitCodeByteCount;

extern uint32_t g_gemm_16bitCode[];
extern size_t g_gemm_16bitCodeByteCount;

extern uint32_t g_gemm_floatCode[];
extern size_t g_gemm_floatCodeByteCount;

uint32_t floatAsUInt32(float input) {
  return *((uint32_t*)&input);
}

void naive_cblas_sgemm(
  int order,
  int transposeA,
  int transposeB,
  int m,
  int n,
  int k,
  float alpha,
  float *a,
  int lda,
  float *b,
  int ldb,
  float beta,
  float* c,
  int ldc) {

  assert((transposeA == JPCblasNoTrans) || (transposeA == JPCblasTrans));
  assert(transposeB == JPCblasNoTrans);
  assert(order == JPCblasColMajor);

  int i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      float total = 0.0f;
      for (l = 0; l < k; l++) {
        int aIndex;
        if (transposeA == JPCblasNoTrans) {
          aIndex = ((lda * l) + i);
        } else {
          aIndex = ((lda * i) + l);
        }
        const float aValue = a[aIndex];
        const int bIndex = ((ldb * j) + l);
        const float bValue = b[bIndex];
        total += (aValue * bValue);
      }
      const int cIndex = ((ldc * j) + i);
      const float oldCValue = c[cIndex];
      c[cIndex] = ((alpha * total) + (beta * oldCValue));
    }
  }
}

void qpu_cblas_sgemm(
  int order,
  int transposeA,
  int transposeB,
  int m,
  int n,
  int k,
  float alpha,
  uint32_t a,
  float aMin,
  float aMax,
  int aBitsPerElement,
  int lda,
  uint32_t b,
  int ldb,
  float beta,
  uint32_t c,
  int ldc) {

  assert(transposeA == JPCblasTrans);
  assert(transposeB == JPCblasNoTrans);
  assert(order == JPCblasColMajor);
  assert(aBitsPerElement == 32);

  struct timeval start;
  gettimeofday(&start, NULL);

  uint32_t* qpuCode = g_gemm_floatCode;
  size_t qpuCodeByteCount = g_gemm_floatCodeByteCount;

  const float aRange = 1.0f;

  if (qpu_enable(1)) {
      fprintf(stderr, "QPU enable failed.\n");
      return;
  }
  printf("QPU enabled.\n");

  const size_t messageByteCount = (NUM_QPUS * NUM_MESSAGE_VALUES * sizeof(uint32_t));

  const int uniformCount = 15;
  const size_t uniformByteCount = (NUM_QPUS * uniformCount * sizeof(uint32_t));

  const int debugCount = 16;
  const size_t debugByteCount = (NUM_QPUS * debugCount * sizeof(float));

  const size_t totalByteCount = (
    messageByteCount +
    qpuCodeByteCount +
    uniformByteCount +
    debugByteCount);

  uint32_t gpuMemoryHandle = mem_alloc(totalByteCount, 4096, GPU_MEM_FLG);
  if (!gpuMemoryHandle) {
    fprintf(stderr, "Unable to allocate %d bytes of GPU memory", totalByteCount);
    return;
  }
  uint32_t gpuMemoryBase = mem_lock(gpuMemoryHandle);
  char* armMemoryBase = (char*)(mapmem(gpuMemoryBase + GPU_MEM_MAP, totalByteCount));

  const size_t messageOffset = 0;
  const size_t codeOffset = (messageOffset + messageByteCount);
  const size_t uniformOffset = (codeOffset + qpuCodeByteCount);
  const size_t debugOffset = (uniformOffset + uniformByteCount);

  char* messageInputArm = (armMemoryBase + messageOffset);
  char* codeInputArm = (armMemoryBase + codeOffset);
  char* uniformInputsArm = (armMemoryBase + uniformOffset);
  char* debugBaseArm = (armMemoryBase + debugOffset);

  memcpy(codeInputArm, qpuCode, qpuCodeByteCount);

  const uint32_t messageInputGpu = (gpuMemoryBase + messageOffset);
  const uint32_t codeInputGpu = (gpuMemoryBase + codeOffset);
  const uint32_t uniformInputsGpu = (gpuMemoryBase + uniformOffset);
  const uint32_t aInputGpu = a;
  const uint32_t bInputGpu = b;
  const uint32_t cInputGpu = c;
  const uint32_t debugBaseGpu = (gpuMemoryBase + debugOffset);

  for (int i=0; i < NUM_QPUS; i++) {

    const size_t currentDebugOffset = (i * debugCount * sizeof(uint32_t));
    uint32_t* currentDebugArm = (uint32_t*)(debugBaseArm + currentDebugOffset);
    for (int index = 0; index < debugCount; index += 1) {
      currentDebugArm[index] = 0xdeadbeef;
    }
    const uint32_t currentDebugGpu = (debugBaseGpu + currentDebugOffset);

    const size_t currentUniformOffset = (i * uniformCount * sizeof(uint32_t));
    uint32_t* currentUniformsArm = (uint32_t*)(uniformInputsArm + currentUniformOffset);
    currentUniformsArm[0] = m;
    currentUniformsArm[1] = n;
    currentUniformsArm[2] = k;
    currentUniformsArm[3] = floatAsUInt32(alpha);
    currentUniformsArm[4] = aInputGpu;
    currentUniformsArm[5] = floatAsUInt32(aMin);
    currentUniformsArm[6] = floatAsUInt32(aRange);
    currentUniformsArm[7] = lda;
    currentUniformsArm[8] = bInputGpu;
    currentUniformsArm[9] = ldb;
    currentUniformsArm[10] = floatAsUInt32(beta);
    currentUniformsArm[11] = cInputGpu;
    currentUniformsArm[12] = ldc;
    currentUniformsArm[13] = currentDebugGpu;
    currentUniformsArm[14] = i;

    const uint32_t uniformsGpu = (uniformInputsGpu + currentUniformOffset);

    const size_t currentMessageOffset = (i * NUM_MESSAGE_VALUES * sizeof(uint32_t));
    uint32_t* messageArm = (uint32_t*)(messageInputArm + currentMessageOffset);
    messageArm[0] = uniformsGpu;
    messageArm[1] = codeInputGpu;
  }

  unsigned ret = execute_qpu(NUM_QPUS, messageInputGpu, 1, 10000);

  // Uncomment this if you want to display the rDebugOutput register value after executing the program
  for (int i=0; i < NUM_QPUS; i++) {
    const size_t currentDebugOffset = (i * debugCount * sizeof(uint32_t));
    uint32_t* currentDebugArm = (uint32_t*)(debugBaseArm + currentDebugOffset);
    for (int index = 0; index < debugCount; index += 1) {
      fprintf(stderr, "%d:%d=%f (0x%08x, %d)\n", i, index,
        *(float*)(&currentDebugArm[index]),
        currentDebugArm[index],
        currentDebugArm[index]);
    }
  }

  unmapmem(armMemoryBase, totalByteCount);
  mem_unlock(gpuMemoryHandle);
  mem_free(gpuMemoryHandle);
  qpu_enable(0);

  struct timeval end;
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;
  long duration = ((seconds) * 1000 + useconds/1000.0) + 0.5;
  fprintf(stderr, "QPU took %ldms\n", duration);

}

void test_gemm() {
  const int inputChannels = 363;
  const int inputHeight = 3025;
  const int outputChannels = 96;
  Buffer* input = new Buffer(Dimensions(inputHeight, inputChannels));
  input->setName("input");
  const bool areWeightsTransposed = true;

  const Dimensions inputDims = input->_dims;
  // We're expecting (# of images, # of values)
  assert(inputDims._length == 2);

  const int imageCount = inputDims[0];
  const int inputValuesCount = inputDims[1];

  const Dimensions weightsDims(outputChannels, inputChannels);
  // We're expecting (# of values in input, # of output channels)
  assert(inputDims._length == 2);
  assert(inputDims._length == 2);
  int inputValuesIndex;
  int outputChannelsIndex;
  if (areWeightsTransposed) {
    inputValuesIndex = 1;
    outputChannelsIndex = 0;
  } else {
    inputValuesIndex = 0;
    outputChannelsIndex = 1;
  }
  assert(weightsDims[inputValuesIndex] == inputValuesCount);

  const Dimensions outputDims(imageCount, outputChannels);
  Buffer* outputCPU = new Buffer(outputDims);
  outputCPU->setName("outputCPU");
  Buffer* outputGPU = new Buffer(outputDims);
  outputGPU->setName("outputGPU");
  Buffer* outputAtlas = new Buffer(outputDims);
  outputAtlas->setName("outputAtlas");

  input->populateWithRandomValues(0, 1);

  const int order = JPCblasColMajor;
  const int transposeA = JPCblasTrans;
  const int transposeB = JPCblasNoTrans;

  const int m = outputChannels;
  const int n = input->_dims[0];
  const int k = input->_dims[1];
  const float alpha = 1.0f;
  const int lda = (transposeA == JPCblasNoTrans) ? m : k;
  const int ldb = k;
  const int ldc = m;
  const float beta = 0.0f;

  const int weightsBitsPerElement = 32;

  Buffer* weights = new Buffer(weightsDims);
  weights->setName("weights");
  weights->populateWithRandomValues(0, 1);

  naive_cblas_sgemm(
    order,
    transposeA,
    transposeB,
    m,
    n,
    k,
    alpha,
    weights->_data,
    lda,
    input->_data,
    ldb,
    beta,
    outputCPU->_data,
    ldc
  );

  qpu_cblas_sgemm(
    order,
    transposeA,
    transposeB,
    m,
    n,
    k,
    alpha,
    weights->_gpuMemoryBase,
    0.0f,
    1.0f,
    32,
    lda,
    input->_gpuMemoryBase,
    ldb,
    beta,
    outputGPU->_gpuMemoryBase,
    ldc
  );

  struct timeval start;
  gettimeofday(&start, NULL);

  cblas_sgemm(
    (CBLAS_ORDER)(order),
    (CBLAS_TRANSPOSE)(transposeA),
    (CBLAS_TRANSPOSE)(transposeB),
    m,
    n,
    k,
    alpha,
    weights->_data,
    lda,
    input->_data,
    ldb,
    beta,
    outputAtlas->_data,
    ldc
  );

  struct timeval end;
  gettimeofday(&end, NULL);
  long seconds  = end.tv_sec  - start.tv_sec;
  long useconds = end.tv_usec - start.tv_usec;
  long duration = ((seconds) * 1000 + useconds/1000.0) + 0.5;
  fprintf(stderr, "ATLAS took %ldms\n", duration);

  buffer_are_all_close(outputCPU, outputGPU);
  buffer_are_all_close(outputAtlas, outputGPU);

  delete weights;
  delete input;
  delete outputCPU;
  delete outputGPU;
  delete outputAtlas;
}

int main(int argc, char **argv) {

  test_gemm();

}
