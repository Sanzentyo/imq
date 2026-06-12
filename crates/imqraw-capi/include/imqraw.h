#ifndef IMQRAW_H
#define IMQRAW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMQRAW_STATUS_MESSAGE_CAP 256

typedef enum ImqrawStatusCode {
  IMQRAW_STATUS_OK = 0,
  IMQRAW_STATUS_NULL_POINTER = 1,
  IMQRAW_STATUS_INVALID_ARGUMENT = 2,
  IMQRAW_STATUS_DECODE_ERROR = 3,
  IMQRAW_STATUS_ENCODE_ERROR = 4
} ImqrawStatusCode;

typedef enum ImqrawPixelFormat {
  IMQRAW_PIXEL_FORMAT_LUMA8 = 1,
  IMQRAW_PIXEL_FORMAT_RGB8 = 2,
  IMQRAW_PIXEL_FORMAT_RGBA8 = 3,
  IMQRAW_PIXEL_FORMAT_BGR8 = 4,
  IMQRAW_PIXEL_FORMAT_BGRA8 = 5,
  IMQRAW_PIXEL_FORMAT_LUMA16LE = 6,
  IMQRAW_PIXEL_FORMAT_RGB16LE = 7,
  IMQRAW_PIXEL_FORMAT_RGBA16LE = 8,
  IMQRAW_PIXEL_FORMAT_RGBF32 = 9,
  IMQRAW_PIXEL_FORMAT_RGBAF32 = 10,
  IMQRAW_PIXEL_FORMAT_YUV444P8 = 11,
  IMQRAW_PIXEL_FORMAT_YUV422P8 = 12,
  IMQRAW_PIXEL_FORMAT_YUV420P8 = 13,
  IMQRAW_PIXEL_FORMAT_NV12 = 14,
  IMQRAW_PIXEL_FORMAT_HSV8 = 15,
  IMQRAW_PIXEL_FORMAT_HSVA8 = 16,
  IMQRAW_PIXEL_FORMAT_BINARY1_LSB = 17,
  IMQRAW_PIXEL_FORMAT_BINARY1_MSB = 18,
  IMQRAW_PIXEL_FORMAT_UNKNOWN = 65535
} ImqrawPixelFormat;

typedef struct ImqrawStatus {
  int32_t code;
  char message[IMQRAW_STATUS_MESSAGE_CAP];
} ImqrawStatus;

typedef struct ImqrawByteView {
  const uint8_t *ptr;
  size_t len;
} ImqrawByteView;

typedef struct ImqrawByteBuffer {
  uint8_t *ptr;
  size_t len;
  size_t cap;
} ImqrawByteBuffer;

typedef struct ImqrawStringView {
  const char *ptr;
  size_t len;
} ImqrawStringView;

typedef struct ImqrawPlaneView {
  const uint8_t *ptr;
  size_t len;
  size_t stride;
} ImqrawPlaneView;

typedef struct ImqrawImageInfo {
  uint32_t width;
  uint32_t height;
  uint16_t pixel_format;
  uint16_t color_space;
  uint16_t transfer;
  uint16_t range;
  size_t plane_count;
} ImqrawImageInfo;

typedef struct ImqrawEncodePlane {
  ImqrawByteView data;
  size_t stride;
} ImqrawEncodePlane;

typedef struct ImqrawEncodeImage {
  uint32_t width;
  uint32_t height;
  uint16_t pixel_format;
  uint16_t color_space;
  uint16_t transfer;
  uint16_t range;
  ImqrawByteView data;
  const ImqrawEncodePlane *planes;
  size_t plane_count;
  ImqrawStringView label;
  const ImqrawStringView *tags;
  size_t tag_count;
} ImqrawEncodeImage;

typedef struct ImqrawBundle ImqrawBundle;

const char *imqraw_version(void);

int32_t imqraw_encode_rgba8(const ImqrawEncodeImage *image,
                            ImqrawByteBuffer *out,
                            ImqrawStatus *status);

int32_t imqraw_encode_luma8(const ImqrawEncodeImage *image,
                            ImqrawByteBuffer *out,
                            ImqrawStatus *status);

int32_t imqraw_encode_image(const ImqrawEncodeImage *image,
                            ImqrawByteBuffer *out,
                            ImqrawStatus *status);

int32_t imqraw_encode_bundle(const ImqrawEncodeImage *images,
                             size_t image_count,
                             ImqrawByteBuffer *out,
                             ImqrawStatus *status);

void imqraw_buffer_free(ImqrawByteBuffer buffer);

int32_t imqraw_bundle_decode(ImqrawByteView bytes,
                             ImqrawBundle **out,
                             ImqrawStatus *status);

void imqraw_bundle_free(ImqrawBundle *bundle);

int32_t imqraw_bundle_image_count(const ImqrawBundle *bundle,
                                  size_t *out,
                                  ImqrawStatus *status);

int32_t imqraw_bundle_image_info(const ImqrawBundle *bundle,
                                 size_t image_index,
                                 ImqrawImageInfo *out,
                                 ImqrawStatus *status);

int32_t imqraw_bundle_image_label(const ImqrawBundle *bundle,
                                  size_t image_index,
                                  ImqrawStringView *out,
                                  ImqrawStatus *status);

int32_t imqraw_bundle_image_tag_count(const ImqrawBundle *bundle,
                                      size_t image_index,
                                      size_t *out,
                                      ImqrawStatus *status);

int32_t imqraw_bundle_image_tag(const ImqrawBundle *bundle,
                                size_t image_index,
                                size_t tag_index,
                                ImqrawStringView *out,
                                ImqrawStatus *status);

int32_t imqraw_bundle_image_plane(const ImqrawBundle *bundle,
                                  size_t image_index,
                                  size_t plane_index,
                                  ImqrawPlaneView *out,
                                  ImqrawStatus *status);

#ifdef __cplusplus
}
#endif

#endif
