#include "imqraw.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static ImqrawStringView sv(const char *value) {
  size_t len = 0;
  while (value[len] != '\0') {
    len += 1;
  }
  return (ImqrawStringView){value, len};
}

static void check(int32_t code, const ImqrawStatus *status, const char *step) {
  if (code == IMQRAW_STATUS_OK) {
    return;
  }
  fprintf(stderr, "%s failed: %s\n", step, status->message);
  exit(1);
}

int main(void) {
  const uint8_t pixels[] = {
      0, 0, 0, 255,
      255, 255, 255, 255,
  };
  const ImqrawStringView tags[] = {sv("reference"), sv("c-api")};
  const ImqrawEncodeImage image = {
      .width = 2,
      .height = 1,
      .pixel_format = IMQRAW_PIXEL_FORMAT_RGBA8,
      .data = {pixels, sizeof pixels},
      .label = sv("c-reference"),
      .tags = tags,
      .tag_count = 2,
  };
  ImqrawStatus status = {0};
  ImqrawByteBuffer bytes = {0};

  check(imqraw_encode_image(&image, &bytes, &status), &status, "encode");
  printf("encoded bytes: %zu\n", bytes.len);

  ImqrawBundle *bundle = NULL;
  check(
      imqraw_bundle_decode((ImqrawByteView){bytes.ptr, bytes.len}, &bundle, &status),
      &status,
      "decode");

  size_t image_count = 0;
  check(imqraw_bundle_image_count(bundle, &image_count, &status), &status, "count");
  printf("images: %zu\n", image_count);

  ImqrawImageInfo info = {0};
  check(imqraw_bundle_image_info(bundle, 0, &info, &status), &status, "info");
  printf("image: %ux%u format=%u planes=%zu\n",
         info.width,
         info.height,
         info.pixel_format,
         info.plane_count);

  ImqrawStringView label = {0};
  check(imqraw_bundle_image_label(bundle, 0, &label, &status), &status, "label");
  printf("label: %.*s\n", (int)label.len, label.ptr);

  ImqrawPlaneView plane = {0};
  check(imqraw_bundle_image_plane(bundle, 0, 0, &plane, &status), &status, "plane");
  printf("plane bytes: %zu stride=%zu first=%u\n", plane.len, plane.stride, plane.ptr[0]);

  imqraw_bundle_free(bundle);
  imqraw_buffer_free(bytes);
  return 0;
}
