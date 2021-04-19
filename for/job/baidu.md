
## 1. Trie树实现前缀查找


```python
class Trie:
    def __init__(self):
        self.children = [None] *26
        self.isEND = False


    def searchPrefix(self, prefix):
        rst = []
        node = self
        for ch in prefix:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node
    
    
    def searchAll(self, prefix):
        node = self.searchPrefix(prefix)
        
        rst = []
        
        if not node:
            return rst
        
        def dfs(node, pth):
            if node.isEND:
                rst.append(pth)
                return 
            choices = []
            for i in range(len(node.children)):
                if node.children[i] is not None:
                    choices.append(chr(i+97))
            for ch in choices:
                dfs(node.children[ord(ch) - ord('a')], pth+ch)
        
        dfs(node, prefix)
        return rst
                

    def insert(self, word):
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEND = True

        
    def build(self, filename):
        with open(filename) as f:
            lines = [x.strip() for x in f.readlines()]
        for word in lines:
            self.insert(word)
        print("Trie builded.")
                
        
    def search(self, word):
        node = self.searchPrefix(word)
        return node is not None and node.isEND


    def startsWith(self, prefix):
        return self.searchPrefix(prefix) is not None

```


```python
##测试
!cat t.lst
```

    apple
    application
    approach
    complete
    company
    computer
    common
    progress
    proactive
    procedure
    


```python
T = Trie()
T.build('t.lst')
```

    Trie builded.
    


```python
T.startsWith('app')
```




    True




```python
T.searchAll('app')
```




    ['apple', 'application', 'approach']




```python
T.searchAll('com')
```




    ['common', 'company', 'complete', 'computer']




```python
T.searchAll('aww')
```




    []



## 2. LRU实现缓存机制


```python
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev  = None
        self.next  = None


class LRUCache:

    def __init__(self, capacity):
        self.cache = dict()
        # 使用伪头部和伪尾部节点    
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key):
        if key not in self.cache:
            return -1
        # 如果 key 存在，先通过哈希表定位，再移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key, value):
        if key not in self.cache:
            # 如果 key 不存在，创建一个新的节点
            node = DLinkedNode(key, value)
            # 添加进哈希表
            self.cache[key] = node
            # 添加至双向链表的头部
            self.addToHead(node)
            self.size += 1
            if self.size > self.capacity:
                # 如果超出容量，删除双向链表的尾部节点
                removed = self.removeTail()
                # 删除哈希表中对应的项
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

```


```python
lru = LRUCache(capacity = 2)
lru.put(1, 1)
lru.put(2, 2)
print([x.value for x in lru.cache.values()])
lru.put(3, 3)
print([x.value for x in lru.cache.values()])
lru.put(4, 4)
print([x.value for x in lru.cache.values()])
lru.put(5, 5)
print([x.value for x in lru.cache.values()])
```

    [1, 2]
    [2, 3]
    [3, 4]
    [4, 5]
    
