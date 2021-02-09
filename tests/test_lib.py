# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import koleksyon.lib as ll


class TestLib(unittest.TestCase):

    def test_find_max_mode(self):
        a = [1,1,2,2,3]
        mm = ll.find_max_mode(a)
        self.assertEqual(mm, 2)



if __name__ == '__main__':
    unittest.main()
