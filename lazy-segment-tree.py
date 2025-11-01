"""
This is a lazy-segment-tree implementation for LeetCode "3721. Longest Balanced Subarray II" 
(https://leetcode.com/problem-list/n6kvtt9j/)

Lazy segment tree represents an array of ints of size `n`.
It supports operations add `delta` to the range `0..i`
Find left-most 0.

It is a binary tree where each node represents a subarray and keeps min and max values in that subarray.
Root node represents the whole array.
Each node except leafs has two children that represents ~halves of the parents subarray.
Leafs represent subarrays of length 1.

It is a lazy segment tree, and that means that when updating the range for nodes representing subarrays fully covered by
update range we do not propagate the changes to the child nodes, but instead just remember the update in the node.

It keeps data in one array "binary heap" style: childs indices = i*2 and i*2+1, array size is N+n, where N is the nearest power of two higher then n.

"""

class LazySegmentTree:
    def __init__(self, n):
        self.N = 1
        self.n = n
        while self.N < n: self.N *= 2
        arr_len = 2*self.N-1
        self.min = [0]*arr_len
        self.max = [0]*arr_len
        self.lazy = [0]*arr_len
        self.first_leaf_index = self.N-1
        # precompute ranges
        self.ranges = [self._compute_range(i) for i in range(arr_len)]

    # range that node at index represents
    # root represents whole array 0..N
    # at each level array is evenly subdivided between children
    # level 0  0:0..N
    # level 1  1:0..N/2, 2:N/2..N
    # level 2  3:0..N/4, 4:N/4..2N/4, 5:2N/4..3N/4, 6:4N/4..N
    # the returned range is [left..right)
    def _compute_range(self, index: int) -> tuple[int,int]:
        level = (index+1).bit_length() - 1
        index0 = (1 << level) - 1
        index_in_level = index - index0
        total_in_level = 1 << level
        left = index_in_level * self.N // total_in_level
        right = (index_in_level + 1) * self.N // total_in_level
        return min(self.n, left), min(self.n, right)

    def _range(self, index: int) -> tuple[int,int]:
        return self.ranges[index]

    # working with indices, if root is 1, then children are index*2+1 and index*2+2, but we want zero based
    def _root(self) -> int:
        return 0
    
    def _children(self, index: int) -> tuple[int, int]:
        return (index+1)*2-1, (index+1)*2        
    
    def _is_leaf(self, index: int) -> bool:
        return index >= self.first_leaf_index

    # should be caled if lazy[index] != 0
    # applies stored lazy updates to children
    def _apply_lazy(self, index: int):
        # update range is 0..node_right, covering whole ranges of children, update value is unapplied previous updates stored in lazy
        left_child, right_child = self._children(index)
        node_left, node_right = self._range(index)
        self._update_range(left_child, node_left, node_right, self.lazy[index])
        self._update_range(right_child, node_left, node_right, self.lazy[index])
        self.lazy[index] = 0

    # apply update to the node at index 
    # update range is [0..right), update is adding delta to all elements in range
    def _update_range(self, index: int, update_left: int, update_right: int, delta: int):
        node_left, node_right = self._range(index)

        if node_right <= update_left or node_left >= update_right:
            # node is not affected by update
            return
        
        if self._is_leaf(index):
            self.min[index] += delta
            self.max[index] += delta
            return
        
        if node_left >= update_left and node_right <= update_right:
            # node is fully in update range
            # just save in lazy
            self.min[index] += delta
            self.max[index] += delta
            self.lazy[index] += delta
            return
        
        # node is partially affected by update
        # pass it to children
        
        # propagate lazy to children
        if self.lazy[index] != 0:
            self._apply_lazy(index)
        
        # propagate update to children
        left_child, right_child = self._children(index)
        mins = []
        maxs = []
        for child in [left_child, right_child]:
            child_l, child_r = self._range(child)
            if child_r-child_l>0:
                self._update_range(child, update_left, update_right, delta)
                mins.append(self.min[child])
                maxs.append(self.max[child])
        self.min[index] = min(mins)
        self.max[index] = max(maxs)

    def apply_all_lazy(self, index = 0):
        if self._is_leaf(index):
            return

        left_child, right_child = self._children(index)

        if self.lazy[index] != 0:
            self._apply_lazy(index)
        
        self.apply_all_lazy(left_child)
        self.apply_all_lazy(right_child)

    def clone(self):
        new_tree = LazySegmentTree(self.n)
        new_tree.min = self.min.copy()
        new_tree.max = self.max.copy()
        new_tree.lazy = self.lazy.copy()
        return new_tree
    
    def get_arr(self) -> list[int]:
        return [self.min[i] for i in range(self.first_leaf_index, self.first_leaf_index+self.n)]


    def print_state(self):
        cur_level = [self._root()]

        while len(cur_level) > 0:
            next_level = []
            # print(cur_level)
            for index in cur_level:
                # print(index)
                print("index", index, "range", self._range(index), "min", self.min[index], "max", self.max[index], "lazy", self.lazy[index])
                if not self._is_leaf(index) and self.lazy[index]==0:
                    left_child, right_child = self._children(index)
                    next_level.append(left_child)
                    next_level.append(right_child)
            cur_level = next_level


    # update the range [0..right), by adding delta to it, delta should be -1 or +1
    def update_range(self, update_left: int, update_right: int, delta: int):
        self._update_range(self._root(), update_left, update_right, delta)

    # find left most leaf with value 0
    def find_left_most_0(self) -> int:
        index = self._root()

        if self.min[index] > 0 or self.max[index] < 0:
            return -1

        while not self._is_leaf(index):
            if self.lazy[index] != 0:
                self._apply_lazy(index)

            left_child, right_child = self._children(index)
            if self.min[left_child] <=0 and self.max[left_child]>=0:
                index = left_child
            else:
                index = right_child

        return index - self.first_leaf_index

import random
import unittest

class TestLazySegmentTree(unittest.TestCase):
    def run_operations(self, ops, n):
        """
        Executes a given sequence of operations on both:
        ✅ brute-force array model
        ✅ LazySegmentTree model
        And validates results after each query.
        """
        st = LazySegmentTree(n)
        arr = [0]*n
        
        def brute_find():
            for i, v in enumerate(arr):
                if v == 0:
                    return i
            return -1

        """
        for find_left_most_0 to work neighbors should not differ by more than 1, so if they do differ more just skip checking
        """
        def is_invariant_held_up() -> bool:
            prev = None
            for i,v in enumerate(arr):
                if prev != None:
                    if abs(v-prev) > 1: return False
                prev = v
            return True

        executed_ops = []

        for (op, *args) in ops:
            executed_ops.append((op, *args))
            if op == "update":
                left, right, delta = args
                # brute array update
                for i in range(left, right):
                    arr[i] += delta
                # tree update
                st.update_range(left, right, delta)

                tree_clone =st.clone()
                tree_clone.apply_all_lazy()
                for index in range(0, len(tree_clone.min)):
                    l, r = tree_clone._range(index)
                    if l>=r: continue
                    if (min(arr[l:r]) != tree_clone.min[index] or max(arr[l:r]) != tree_clone.max[index]):
                        print(l, r)
                        print("❌ MISMATCH in tree state")
                        print("Operations log:")
                        for o in executed_ops: print(" ", o)
                        print("Ref  Array:", arr)
                        print("Tree Array:", tree_clone.get_arr())
                        print("Tree state:")
                        st.print_state()
                        self.fail("Mismatch")


            elif op == "query":
                expected = brute_find()
                result = st.find_left_most_0()
                if is_invariant_held_up() and expected != result:
                    tree_clone =st.clone()
                    tree_clone.apply_all_lazy()
                    print("❌ MISMATCH FOUND")
                    print("Operations log:")
                    for o in executed_ops: print(" ", o)
                    print("Tree result:", result, "Expected:", expected)
                    print("Ref. Array:", arr)
                    print("Tree Array:", tree_clone.get_arr())
                    print("Tree state:")
                    st.print_state()
                    self.fail("Mismatch")

        return True

    def random_test_case(self, seed, num_ops=10, n=20):
        """
        Generates a random sequence then passes it to run_operations().
        """
        print(f"\n🔹 Running seed = {seed}")
        random.seed(seed)

        ops = []
        for _ in range(num_ops):
            if random.random() < 0.6:  # ~60% updates, 40% queries
                right = random.randint(0, n)
                left = random.randint(0, right)
                delta = random.choice([-1, +1])
                ops.append(("update", left, right, delta))
            else:
                ops.append(("query",))

        return self.run_operations(ops, n)
    
    def test_simple(self):
        self.run_operations([
            ('update', 0, 7, -1),
            ("query"),
            ], 
            20)

    def test_random(self):
        # many short independent cases
        for seed in range(1, 500):
            self.random_test_case(seed)

if __name__ == "__main__":
    unittest.main()