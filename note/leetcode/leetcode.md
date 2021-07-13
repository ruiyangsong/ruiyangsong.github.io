

# 

1. 递归问题，画出递归树分析计算量和计算顺序

   <img src=".images/image-20210629100257833.png" alt="image-20210629100257833" style="zoom:23%;" />

2. 



# 1. 排序

<img src=".images/cde64bf682850738153e6c76dd3f6fb32201ce3c73c23415451da1eead9eb7cb-20190624173156.jpg" alt="20190624173156.jpg" style="zoom:60%;" />

### 1.1 冒泡

**思路**：步长为1，窗口为2依次滑动，将每一对的较小值放在前面

一次滑动完成之后，最大值放在了最后，下一步处理`nums[:, -1]`，故需要处理 n-1 次

算法终止条件：一次扫描中没有发生交换

```python
def bubble_sort(nums):
    for i in range(1, len(nums)):
        flag = False
        for j in range(len(nums)-i):
            if nums[j + 1] < nums[j]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                flag = True
        if not flag:
            break
    print(nums)
```

## 1.2 快排

**思路**：冒泡排序基础上的==递归分治法==，冒泡每次相邻的交换，快排打破了这个限制

1. 从数列中挑出一个元素，称为 "基准"（pivot）;
2. 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
3. 递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序，所以partition函数中有index来标识；

```python
def quick_sort(nums, left, right):
    if left < right:
        index = partition(nums, left, right)
        quick_sort(nums, left, index-1)
        quick_sort(nums, index+1, right)
    print(nums)

def partition(nums, left, right):
    pivot = left
    index = pivot + 1
    i = pivot + 1
    while i <= right:
        if nums[i] < nums[pivot]:
            nums[i], nums[index] = nums[index], nums[i]
            index += 1
        i += 1
    nums[index-1], nums[pivot] = nums[pivot], nums[index-1]
    return index-1
```



## 1.3 归并排序

```python
def merge_sort(nums):
    if len(nums) < 2:
        return nums
    mid = len(nums) // 2
    left_part  = nums[:mid]
    right_part = nums[mid:]
    return merge(merge_sort(left_part), merge_sort(right_part))

def merge(left_part, right_part):
    rst = []
    while left_part and right_part:
        if left_part[0] >= right_part[0]:
            rst.append(right_part.pop(0))
        else:
            rst.append(left_part.pop(0))
    if left_part:
        rst.extend(left_part)
    elif right_part:
        rst.extend(right_part)
    return rst
```



## 1.4 堆排序

1. 创建一个堆 H[0……n-1]；
2. 把堆首（最大值）和堆尾互换；
3. 把堆的尺寸缩小 1，并调用 shift_down(0)，目的是把新的数组顶端数据调整到相应位置；
4. 重复步骤 2，直到堆的尺寸为 1。

```python
def heapify(nums, i, len_max=None):
    if len_max is None:
        len_max = len(nums)

    left = 2 * i + 1
    right = 2 * i + 2
    largest_idx = i
    if left < len_max and nums[left] > nums[largest_idx]:
        largest_idx = left
    if right < len_max and nums[right] > nums[largest_idx]:
        largest_idx = right
    if largest_idx != i:
        nums[i], nums[largest_idx] = nums[largest_idx], nums[i]
        heapify(nums, largest_idx, len_max)


def build_heap(nums):
    '''
    :param nums: [1, 2, 3, 5, 1]
    :return:     [5, 2, 3, 1, 1]
    '''
    for i in range(len(nums)//2, -1, -1):
        heapify(nums, i)

    return nums


def heap_sort(nums):
    len_max = len(nums)
    nums = build_heap(nums)
    for i in range(len(nums)-1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]

        len_max -= 1
        heapify(nums, 0, len_max)

    return nums
```



## 双排序













































# 1. 动态规划

## 概述

1. 适合的场景

   有重叠子问题、具有最优子结构的场景，通过 dp table 来优化穷举过程

2. 套路

   ```
   套路：
   1. 确定状态和选择
   2. 明确dp数组含义，用dp数组把状态描述出来
   3. 定义 base case
   4. 状态转移方程(根据选择确定状态转移)
   ```

   ```python
   for 状态1 in 状态1列表:
   	for 状态2 in 状态2列表:
         	for ...
          		dp[状态1][状态2][...] = 择优(选择1, 选择2, ...)
   ```

3. 一些理解

   - dp 通过备忘录来优化穷举，其是自底向上由base case计算后续结果，区别于递归（自顶向下），因此dp一般用循环而不是递归。

   - 数组的遍历方向

     * 遍历的过程中，所需的状态必须是已经计算出来的

     * 遍历的终点必须是存储结果的那个位置。

       

## 零钱兑换

### [零钱兑换](https://leetcode-cn.com/problems/coin-change/)

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的**最少的硬币个数**。如果没有任何一种硬币组合能组成总金额，返回 -1。每种硬币的数量是无限的。



1. 状态：也就是原问题和子问题中变化的变量。由于硬币数量无限，所以唯一的状态就是目标金额`amount`。

   选择：从硬币列表选择硬币，然后amount减小

2. dp数组是一维的, dp[i] = 金额为i时，需要的最少硬币数量

3. base case：目标金额为0时，所需硬币数量为0

4. 状态转移方程

   dp[0] = 0

   dp[n] = -1, n<0 

   dp[n] = min{dp[n-coin] + 1 for each coin}, n>0

```python
def coinChange(self, coins: List[int], amount: int) -> int:
  dp = [float('inf')] * (amount+1)
  dp[0] = 0
  for i in range(1, amount+1):
    for coin in coins:
      if i >= coin:
	      dp[i] = min(dp[i], dp[i-coin] + 1)
	return dp[-1] if dp[-1] != float('inf') else -1 ## 考虑凑不出的情形返回-1
```

### [零钱兑换2](https://leetcode-cn.com/problems/coin-change-2/)

返回可以凑成总金额的 **所有组合方法的个数**。如果任何硬币组合都无法凑出总金额，返回 `0` 

1. 状态：总金额

    选择：选择硬币，总金额减小

2. dp[i]：能够装满i的所有方式

3. Base case: dp[0] = 1, 总金额为0时，凑法为1

4. 状态转移

    dp[i] = dp[i] + dp[i-coin],  for each coin in coins

    先循环coin再循环[coin,amount]可以保证不出现组合数的重复，可以这样理解：

    对于一个coin，对于coin 到 amount之间的数i，dp[i] 增加 dp[i-coin]，循环每个coin可得到最终结果

```python
def change(self, amount: int, coins: List[int]) -> int:
  dp = [0] * (amount+1)
  dp[0] = 1
  # 先遍历coin再遍历amount，避免重复计算组合（如，1，2和2，1是同一种）
  for coin in coins:
    for i in range(coin, amount+1):
  ##for i in range(amount+1):
    ##for coin in coins:
      if i >= coin:
        dp[i] += dp[i-coin]
  return dp[-1]
```

## 子序列问题

### ==子序列问题模版==



### 编辑距离（2维dp）

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符

dp含义：`dp[i-1][j-1]` 存储 `s1[0..i]` 和 `s2[0..j]` 的最小编辑距离

<img src=".images/image-20210709005845106.png" alt="image-20210709005845106" style="zoom:25%;" />



```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1] + 1, # 替换
                        dp[i - 1][j] + 1,     # 把i对应的字符删掉
                        dp[i][j - 1] + 1      # 在i位置插入一个字符与j匹配
                    )
        return dp[-1][-1]
```



### 最长递增子序列（1维dp）

#### dp解法

给一个无序的整数数组，求其中最长上升子序列的长度

```
输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
```

1. 状态是序列的每一个字符，dp是一维的

2. **dp[i] 表示以 nums[i] 这个数结尾的最长递增子序列的长度**。==为什么不能是前i个中最长递增子序列的长度==

   返回dp的最大值

3. base case: 每个字符至少为1，dp全部初始化为1

4. 状态转移

   对于`nums[i]`,查找0-i中小于nums[i]的位置j，计算dp[j]+1中的最大值作为dp[i]的值

```python
def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return None
        dp = [1] * len(nums)
        for i in range(len(nums)):
            lst = [dp[j] for j in range(i) if nums[j] < nums[i]]
            if lst:
                dp[i] = max(lst) + 1
                
        return max(dp)
```

#### 二分查找

**patience sort**

只能把点数小的牌压到点数比它大的牌上；如果当前牌点数较大没有可以放置的堆，则新建一个堆，把这张牌放进去；如果当前牌有多个堆可供选择，则选择最左边的那一堆放置。

这样保证top cards有序，堆的个数就是最长递增子序列的长度

<img src=".images/image-20210710082834712.png" alt="image-20210710082834712" style="zoom:50%;" />

我们只要把处理扑克牌的过程写出来即可。每次处理一张扑克牌要找一个合适的牌堆顶来放，牌堆顶的牌**有序**，可用二分查找来搜索当前牌应放置的位置。

```python
def lengthOfLIS(nums):
    n = len(nums)
    top = [0] * n
    # 牌堆数初始化为 0
    piles = 0
    for i in range(n):
        # 对每张牌，查找放置位置
        poker = nums[i]
        # 搜索左侧边界的二分查找
        left = 0
        right = piles
        while (left < right):
            mid = (left + right) // 2
            if top[mid] >= poker:
                right = mid
            else top[mid] < poker:
                left = mid + 1
        # 没找到合适的牌堆，新建一堆
        if left == piles: 
            piles+=1
        # 把这张牌放到牌堆顶
        top[left] = poker
    #牌堆数就是 LIS 长度
    return piles
```



### 二维递增子序列：信封嵌套

给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。请计算最多能有多少个信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

**思路**：标准的 LIS 算法只能在数组中寻找最长子序列，而信封是由`(w,h)`这样的二维数对形式表示的，如何把 LIS 算法运用过来呢？固定一个维度，再在另一个维度上进行选择：先对w这一列升序排序，对于w相同的再对h逆序排序。这个解法的关键在于，对于宽度`w`相同的数对，要对其高度`h`进行降序排序。因为两个宽度相同的信封不能相互包含，而逆序排序保证在`w`相同的数对中最多只选取一个计入 LIS。

```python
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        ## 对第一列进行正向排序，再对第二列进行逆向排序
        if not envelopes:
            return 0
        
        n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0], -x[1]))

        f = [1] * n
        for i in range(n):
            for j in range(i):
                if envelopes[j][1] < envelopes[i][1]:
                    f[i] = max(f[i], f[j] + 1)
        
        return max(f)
```





### 最大子数组和

给定一个整数数组 `nums` ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**状态：**以`nums[i]`为结尾的「最大子数组和」为`dp[i]`。

**选择**：假设`dp[i-1]`已知，`dp[i]`有两种「选择」，要么与前面的相邻子数组连接，形成一个和更大的子数组；要么不与前面的子数组连接，自成一派，自己作为一个子数组。

```python
def max_sub_arr(nums):
    n = len(nums)
    dp = nums.copy()
    for i in range(1, n):
        dp[i] = max(dp[i-1] + nums[i], nums[i])
    return max(dp)
```

#### 状态压缩

dp[i]只和dp[i-1]有关

```python
def max_sub_arr(nums):
    n = len(nums)
    dp0 = nums[0]
    res = nums[0]
    for i in range(1, n):
        dp0 = max(dp0 + nums[i], nums[i])
        res = max(res, dp0)
    return res
```

### 最长公共子序列（2维dp）

输入: str1 = "abcde", str2 = "ace" 
输出: 3 
解释: 最长公共子序列是 "ace"，它的长度是 3

`子序列类型的问题，穷举出所有可能的结果都不容易，而动态规划算法做的就是穷举 + 剪枝，它俩天生一对儿。所以可以说只要涉及子序列问题，十有八九都需要动态规划来解决`

1. 明确 dp table

   一般两个字符串的问题，都是构造一个`m+1 * n+1`的dp table

   <img src=".images/image-20210630004525108.png" alt="image-20210630004525108" style="zoom:33%;" />

   **`dp[i][j]`的含义是：对于`s1`中前`i`个和`s2`中前`j`个，它们的 LCS 长度是`dp[i][j]`**

2. Base case

   `dp[0][..]`和`dp[..][0]`都为0

3. 选择：`i,j`对应的字符在不在LCS中

4. 状态转移

   如果`s1[i]==s2[j]`则这个字符一定在LCS中， `dp[i][j] = dp[i-1][j-1]+1`

   如果`s1[i]!=s2[j]`则至少有一个字符不在LCS中， `dp[i][j] = max(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])`

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
```



#### 两个字符串的删除操作

给定两个单词 *word1* 和 *word2*，找到使得 *word1* 和 *word2* 相同所需的最小步数，每步可以删除**任意一个字符串**中的**任意一个字符**。

**思路**：删除的结果就是这两个字串的最长公共子序列，删除的次数可以通过最长公共子序列的长度推导出来

```python
def minDistance(s1, s2):
    m = len(s1)
    n = len(s2)
    lcs = longestCommonSubsequence(s1, s2)
    return m - lcs + n - lcs
```

#### [两个字符串的最小ASCII删除和](https://leetcode-cn.com/problems/minimum-ascii-delete-sum-for-two-strings/)

给定两个字符串`s1, s2`，找到使两个字符串相等所需删除字符的ASCII值的最小和。

**思路**：不要求公共子序列最长，不能直接复用计算最长公共子序列的函数了，但是可以依照之前的思路，**稍微修改 base case 和状态转移部分即可直接写出解法代码**：

```python
## 将 s1[i..] 和 s2[j..] 删除成相同字符串, 最小的 ASCII 码之和为 dp(i, j)
def fun_dp(s1,s2):
    m, n = len(s1), len(s2)
    dp = [[0] for _ in range(n+1) for _ in range(m+1)]
    for i in range(1, m):
        for j in range(1, n):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
def minimumDeleteSum(self, s1: str, s2: str) -> int:
    
    
```











### 最长回文子序列==（2维dp）==

1. dp 数组的定义是：**在子串`s[i..j]`中，最长回文子序列的长度为`dp[i][j]`**

   ==注意`i>=j`==

假设已知`dp[i+1][j-1]`,计算`dp[i][j]`

- `s[i]==s[j] ==> dp[i][j]=dp[i+1][j-1]+2`

- `s[i]!=s[j]`说明s[i]、 s[j]不同时出现在以s[i..j]为子串的最长回文子序列中

  那么`dp[i][j]`等于它俩**分别**加入`s[i+1..j-1]`中，看看哪个子串产生的回文子序列更长

  即：`dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])`

最终要求的是`dp[0][n-1]`

2. Base case

   对角线元素为1

   `i>j`即左下角初始化为0

   <img src=".images/image-20210702101659140.png" alt="image-20210702101659140" style="zoom:33%;" />

3. 遍历原则

   根据依赖关系选择和最终求解的位置选择遍历方式，如dp计算时依赖左边，下边和左下的元素，最终停留在求解的位置左上角。

   <img src=".images/image-20210702102016416.png" alt="image-20210702102016416" style="zoom:33%;" />

```python
def longestPalindromeSubseq(s):
    n = len(s)
    # dp 数组全部初始化为 0
    dp = [[0 for _ in range(n)] for _ in range(n)]
    # base case
    for i in range(n):
        dp[i][i] = 1
    #反着遍历保证正确的状态转移,先计算右下角，从下往上 从左往右 遍历
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            #状态转移方程
            if s[i] == s[j]
                dp[i][j] = dp[i + 1][j - 1] + 2
            else
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
    #整个 s 的最长回文子串长度
    return dp[0][n - 1]
```

#### **状态压缩（将二维数组 ==投影== 到一维数组）**

计算`dp[i][j]`只需要这三个相邻状态，其实根本不需要维护那么大一个二维的 dp table.

<img src=".images/image-20210702105649594.png" alt="image-20210702105649594" style="zoom:33%;" />

向下投影时（一般向下投影，把`i`这个维度去掉），`dp[i][j-1]` 和 `dp[i+1][j-1]` 其中有一个值会被覆盖，我们采用一个临时变量保存来解决这个问题。

先无脑丢掉维度`i`

**站在当前`i`和`j`的位置分析`dp[j]`和`dp[j-1]`：**

1. `dp[i+1][j]`： `j`循环更改一维dp数组`dp[j]`的值，对于一个固定的$j$，在更改其值之前，其存储的是上一次修改是在外层循环`i=i+1`时的值，即`dp[j]`对应原始二维dp数组的`dp[i+1][j]`

2. `dp[i][j-1]`：  `dp[j-1]`是在内层循环j中的上一次（j-1）进行修改的，此时的外层循环为i，即`dp[j-1]`对应原始数组`dp[i][j-1]`

   

3. `dp[i+1][j-1]`：目前原始数组`dp[i+1][j] ==> dp[j]`, `dp[i][j-1] ==> dp[j-1]`，还剩下`dp[i+1][j-1]`在一维dp数组中没有对应的位置，因此，我们定义一个临时变量存储每次变化的`dp[i+1][j-1]`。 **那么如果我们想得到`dp[i+1][j-1]`，就必须在它被覆盖之前用一个临时变量`temp`把它存起来，并把这个变量的值保留到计算`dp[i][j]`的时候**

4. Base case：将二维base case投影成一维base case

```python
def longestPalindromeSubseq(s):
    n = len(s)
    # dp 数组全部初始化为 0
    #dp = [[0 for _ in range(n)] for _ in range(n)]
    # base case
    #for i in range(n):
    #    dp[i][i] = 1
    dp = [1]*len(n)
    #反着遍历保证正确的状态转移,先计算右下角，从下往上从左往右便利
    for i in range(n - 2, -1, -1):
        pre = 0
        for j in range(i + 1, n):
            tmp = dp[j]
            #状态转移方程
            if s[i] == s[j]
                #dp[i][j] = dp[i + 1][j - 1] + 2
              	dp[j] =pre + 2
            else
                #dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
              	dp[j] = max(dp[j], dp[j-1])
        	pre = tmp
    #整个 s 的最长回文子串长度
    return dp[0][n - 1]
```

#### 最长回文子串（字串是连续的）

首先注意的点：回文串可能是奇数也可能是偶数，一般用**双指针**来解决，从中间向两边扩展

```python
def longestPalindrome(self, s: str) -> str:
    ## 	中心扩展
    def find_max(s, l, r):
      while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r+=1
        return l+1, r-1
    
    ## 每个位置都进行中心扩展
    top, down = 0,0
    for i in range(len(s)):
      ## 以i为中心的回文子串
      l1, r1 = find_max(s,i,i)
      if r1-l1 > down-top:
        top, down = l1, r1
      ## 以i和i+1为中心的回文子串
      l2, r2 = find_max(s,i,i+1)
      if r2-l2 > down-top:
        top, down = l2, r2
return s[top: down+1]
```





## 下降路径最小和

给你一个 `n x n` 的 **方形** 整数数组 `matrix` ，请你找出并返回通过 `matrix` 的**下降路径** 的 **最小和**

```python
	def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        m,n = len(matrix), len(matrix[0])
        dp = [[0 for _ in range(n)] for _ in range(m)] #到matrix[i][j]的最小路径和
        ## 第一行初始化为本身
        for j in range(n):
            dp[0][j] = matrix[0][j]    
        for i in range(1,m):
            for j in range(n):
                if j == n-1:
                    dp[i][j] = matrix[i][j] + min(
                    dp[i-1][j],
                    dp[i-1][j-1]
                    )
                elif j == 0:
                    dp[i][j] = matrix[i][j] + min(
                    dp[i-1][j],
                    dp[i-1][j+1]
                    )
                else:
                    dp[i][j] = matrix[i][j] + min(
                        dp[i-1][j],
                        dp[i-1][j-1],
                        dp[i-1][j+1]
                        )
        return min(dp[-1])
```







## 背包问题

给定一个target（背包容量）和一个数组（物品），能否按照一定方式选取物品得到target

选取方式：每个元素选一次/每个元素选多次/选元素进行排列组合

### 01背包问题

给你一个可装载重量为`W`的背包和`N`个物品，每个物品有重量和价值两个属性。其中第`i`个物品的重量为`wt[i]`，价值为`val[i]`，现在让你用这个背包装物品，最多能装的价值是多少？

1.  明确状态和选择

   状态：用来描述一个问题所需要的变量，即背包容量和可选择的物品，所以状态有两个背包容量和物品重量

   选择：物品装或者不装

2. ==用dp描述状态==

   状态有两个，用二维数组描述

   `dp[i][w]`：对于前`i`个物品，当前背包的容量为`w`，这种情况下可以装的最大价值是`dp[i][w]`

   所求的是`dp[N][W]`

3. base case

   `dp[...][0]` = 0, `dp[0][...]`=0

   ```python
   dp = [N+1][W+1]
   dp[0][...] = 0
   dp[...][0] = 0
   
   for i in [1..N]:
     for w in [1..W]:
       dp[i][w] = max(
         物品i装进背包，
         物品i不装进背包
       )
   return dp[N][W]
   ```

4. 根据==选择==确定状态转移方式

   如果第`i`个物品没有装进背包：`dp[i][w]=dp[i-1][w]`

   如果第`i`个物品没有装进背包：`dp[i][w]=dp[i-1][w-wt[i-1]]+val[i-1]`

   迭代结果是上述两个的最大值

   ```python
   for i in [1..N]:
       for w in [1..W]:
         if w - wt[i-1]<0:
           dp[i][w] = dp[i-1][w] # 背包装不下第i个物品
         else:
           dp[i][w] = max(
             dp[i-1][w],
             dp[i-1][w - wt[i-1]] + val[i-1]
           )
   return dp[N][W]
   ```

   

### 分割等和子集

给一个容量为`sum/2`的背包和一个物品列表（物品只包含重量），问是否存在一种装法刚好装满背包

1. 状态和选择

   状态：描述问题需要背包容量和物品重量，故状态是背包容量和物品重量

   选择：物品装或不装

2. 用dp描述状态

   **`dp[i][w]`：前`i`个商品是否能够装满容量为`w`的背包**

   最终求的是`dp[i][sum/2]`

3. base case

   `dp[0][...]`=False	#没有物品时不能装满背包

   `dp[...][0]`=True     #背包容量为0时可以什么都不装

4. 根据==选择==确定状态转移

   第`i`个商品装进背包：`dp[i][w] = dp[i-1][w-nums[i-1]]`

   第`i`个商品不装进被背包：`dp[i][w] = dp[i-1][w]`

   上述两个选择只有有一个是True，则此步迭代结果就是True

注意的点：

1. 如果sum是奇数，肯定不能满足条件
2. 判断`w-nums[i-1]`是否小于0
3. 注意index的对应关系

```python
def canPartition(self, nums: List[int]) -> bool:
    # 如果和为奇数，不满足条件
    s = sum(nums)
    if s & 1:
        return False
    # 初始化
    w = s//2 #背包容量
    dp = [[False for _ in range(w+1)] for _ in range(len(nums)+1)]
    for i in range(len(nums)+1):
        dp[i][0] = True

    # 状态转移
    for i in range(1, len(nums)+1):
        for j in range(1, w+1):
            dp[i][j] = dp[i-1][j]
            if nums[i-1] <= j:
                dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
	return dp[-1][-1]
```



#### 状态压缩

观察dp数组的转移方式，`dp[i][j]`依赖的是上一行`dp[i-1][...]`转移过来的。

<img src=".images/image-20210705104514281.png" alt="image-20210705104514281" style="zoom:50%;" />

我们直接将维度`i`去掉，向下投影，此时需要注意的是，`dp[i][j]`在更新过程中覆盖`dp[i-1][j]`，此时更新`dp[i][j+1]`的时候无法利用原始的`dp[i-1][j]`，因此状态压缩之后我们反向遍历。

```python
def canPartition(self, nums: List[int]) -> bool:
    # 如果和为奇数，不满足条件
    s = sum(nums)
    if s & 1:
        return False
    # 初始化
    w = s//2 #背包容量
    dp = [False for _ in range(w+1)]
    dp[0] = True # 背包容量为0时的填充方式
    # 状态转移
    for i in range(1, len(nums)+1):
        for j in range(w, 0, -1):
            if nums[i-1] <= j:
                dp[j] = dp[j] or dp[j-nums[i-1]]
	return dp[-1]
```







### 组合背包问题



### 分组背包问题





















# 回溯

## 概述

**回溯算法不像动态规划存在重叠子问题可以优化，回溯算法就是纯暴力穷举，复杂度一般都很高**。

1. 套路

   回溯问题是一个决策树的遍历过程

   1. 路径：已经做出的选择
   2. 选择列表：当前可以做出的选择
   3. 结束条件：到达决策树底层，无法再做出选择

2. 算法模版

```python
rst = []
def bashtrack(pth, choice_lst):
    if 满足结束条件:
        rst.append(pth)
        return 
    for choice in choice_lst:
        做选择
        backtrack(pth_new, choice_lst_new) #此处应该创建新的pth和choice_lst,否则影响下一次循环
```

## 1. 全排列

给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

```python
def permute(self, nums: List[int]) -> List[List[int]]:
    rst = []
    def backtrack(nums, pth):
        if not nums:
            rst.append(pth)
            return
        for i in range(len(nums)):
            backtrack(nums[:i]+nums[i+1:], pth+[nums[i]])
    backtrack(nums, [])
    return rst
```

## 2. [N皇后问题](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247484709&idx=1&sn=1c24a5c41a5a255000532e83f38f2ce4&chksm=9bd7fb2daca0723be888b30345e2c5e64649fc31a00b05c27a0843f349e2dd9363338d0dac61&scene=21#wechat_redirect)

给一个 N×N 的棋盘，放置 N 个皇后，使得它们不能互相攻击。

PS：皇后可以攻击同一行、同一列、左上左下右上右下四个方向的任意单位。

这是 N = 8 的一种放置方法：

<img src=".images/image-20210707225447442.png" alt="image-20210707225447442" style="zoom:33%;" />

问题本质上跟全排列问题差不多，决策树的每一层表示棋盘上的每一行；每个节点可以做出的选择是在该行的任意一列放置一个皇后。

## 3. 目标和

### 回溯解法

i为index，sums为当前和

实际上是一个二叉树的遍历问题，树的深度为nums的长度，故时间复杂度是`O(2^N)`

```python
### 时间超出限制 (回溯)
def helper(i, sums):
    '''sums = sum(nums[:i])'''
    if i == len(nums):
        if sums == target:
            return 1
        else:
            return 0
    else:
        return helper(i+1, sums+nums[i]) + helper(i+1, sums-nums[i])

    return helper(0, 0)
```

### 回溯剪枝

回溯解法中有重叠子问题，如当`nums[i]=0`时，

$helper(nums, i+1, sums+nums[i], target)=helper(nums, i+1, sums-nums[i], target)$是重叠子问题。

思路：通过备忘录记忆,主键是（i，sums）

```python
### 回溯 + memo
memo = {}
def helper(i, sums):
    if i == len(nums):
        if sums == target:
            return 1
        else:
            return 0
    else:
        k = f'{i},{sums}'
        if k in memo:
            return memo[k]
        else:
            rst = helper(i+1, sums+nums[i]) + helper(i+1, sums-nums[i])
            memo[k] = rst
            return rst
return helper(0,0)
```

### dp解法（分割等和子集）

**分析：**我们把nums分成两个子集 `A` 和 `B`，分别代表分配 `+` 的数和分配 `-` 的原始数字，那么他们和 `target` 存在如下关系：

```
sum(A) - sum(B) = target
sum(A) = target + sum(B)
sum(A) + sum(A) = target + sum(B) + sum(A)
2 * sum(A) = target + sum(nums) //将等式右边凑成一个常数值
sum(A) = (target + sum(nums)) / 2
```

此时问题变为一个分割等和子集问题

```python
def findTargetSumWays(self, nums: List[int], S: int) -> int:
	sumAll = sum(nums)
    if S > sumAll or (S + sumAll) % 2:
    	return 0
    target = (S + sumAll) // 2
    dp = [0] * (target + 1)
    dp[0] = 1

    for num in nums:
    	for j in range(target, num - 1, -1):
        	dp[j] = dp[j] + dp[j - num]
    return dp[-1]
```



# 滑动窗口

滑动窗口算法无非就是双指针形成的窗口扫描整个数组/子串，但关键是，你得清楚地知道什么时候应该移动右侧指针来扩大窗口，什么时候移动左侧指针来减小窗口。