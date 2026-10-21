import unittest

from blog.posts import build_index
from blog.slug import slugify


class PostsTest(unittest.TestCase):
    def test_ascii(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_duplicates(self):
        idx = build_index([{"title": "Hello World"}, {"title": "hello world"}, {"title": "Hello-World"}])
        self.assertEqual([s for s, _ in idx], ["hello-world", "hello-world-2", "hello-world-3"])


if __name__ == "__main__":
    unittest.main()
