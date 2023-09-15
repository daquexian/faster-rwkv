#include "tokenizer.h"

#include <gtest/gtest.h>

TEST(WorldTokenizer, load) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));

  rwkv::WorldTokenizer tokenizer(model_dir + "/world_tokenizer");
}

TEST(WorldTokenizer, encode_decode) {
  const std::string model_dir(std::getenv("FR_MODEL_DIR"));

  rwkv::WorldTokenizer tokenizer(model_dir + "/world_tokenizer");
  auto ids = tokenizer.encode("今天天气不错");
  EXPECT_EQ(ids.size(), 6);
  EXPECT_EQ(ids[0], 10381);
  EXPECT_EQ(ids[1], 11639);
  EXPECT_EQ(ids[2], 11639);
  EXPECT_EQ(ids[3], 13655);
  EXPECT_EQ(ids[4], 10260);
  EXPECT_EQ(ids[5], 17631);

  auto str = tokenizer.decode(ids);
  EXPECT_EQ(str, "今天天气不错");
}

TEST(ABCTokenizer, encode_decode) {
  rwkv::ABCTokenizer tokenizer;
  auto ids = tokenizer.encode("S:2");
  EXPECT_EQ(ids.size(), 3);
  EXPECT_EQ(ids[0], 83);
  EXPECT_EQ(ids[1], 58);
  EXPECT_EQ(ids[2], 50);

  auto str = tokenizer.decode(ids);
  EXPECT_EQ(str, "S:2");
}

