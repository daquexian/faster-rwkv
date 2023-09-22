#include <tokenizer.h>

#include <gtest/gtest.h>

#include <tests/utils.h>

using namespace rwkv;

TEST(Tokenizer, load) {
  const std::string filename = TEST_FILE("world_tokenizer");

  Tokenizer tokenizer(filename);
}

TEST(Tokenizer, encode_decode) {
  const std::string filename = TEST_FILE("world_tokenizer");

  Tokenizer tokenizer(filename);
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

TEST(OldTokenizer, encode_decode) {
  const std::string filename = TEST_FILE("old_world_tokenizer");

  Tokenizer tokenizer(filename);
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

TEST(OldABCTokenizer, encode_decode) {
  ABCTokenizer tokenizer;
  auto ids = tokenizer.encode("S:2");
  EXPECT_EQ(ids.size(), 3);
  EXPECT_EQ(ids[0], 83);
  EXPECT_EQ(ids[1], 58);
  EXPECT_EQ(ids[2], 50);

  auto str = tokenizer.decode(ids);
  EXPECT_EQ(str, "S:2");
}

TEST(ABCTokenizerFromEmptyPath, encode_decode) {
  Tokenizer tokenizer("");
  auto ids = tokenizer.encode("S:2");
  EXPECT_EQ(ids.size(), 3);
  EXPECT_EQ(ids[0], 83);
  EXPECT_EQ(ids[1], 58);
  EXPECT_EQ(ids[2], 50);

  auto str = tokenizer.decode(ids);
  EXPECT_EQ(str, "S:2");
}

TEST(SimpleTokenizer, encode_decode) {
  const std::string filename = TEST_FILE("simple_abc_tokenizer");

  Tokenizer tokenizer(filename);
  auto ids = tokenizer.encode("S:2");
  EXPECT_EQ(ids.size(), 3);
  EXPECT_EQ(ids[0], 52);
  EXPECT_EQ(ids[1], 27);
  EXPECT_EQ(ids[2], 19);

  auto str = tokenizer.decode(ids);
  EXPECT_EQ(str, "S:2");
}


TEST(NewABCTokenizer, encode_decode) {
  const std::string filename = TEST_FILE("abc_tokenizer_v20230913");

  Tokenizer tokenizer(filename);
  auto ids = tokenizer.encode("S:2");
  EXPECT_EQ(ids.size(), 3);
  EXPECT_EQ(ids[0], 52);
  EXPECT_EQ(ids[1], 27);
  EXPECT_EQ(ids[2], 19);

  auto str = tokenizer.decode(ids);
  EXPECT_EQ(str, "S:2");
}

