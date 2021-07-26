# Note

## 代码规范
### 变量名称
1. 常量名全部大写
2. 模块中的私有变量/函数：外部不需要引用的函数全部定义成private（函数名以下划线开始），只有外部需要引用的函数才定义为public。只是一种习惯，private其实也可以外部直接调用。  


## 对象和变量
```
python的所有数据类型/容器/函数皆对象: id() --> CPython 中 id() 函数用于获取对象的内存地址

变量/函数名 --> 数值类型对象/函数对象的“引用”
完全可以把函数名赋给一个变量，相当于给这个函数起了一个“别名”

对于不变对象来说，调用对象自身的任意方法，也不会改变该对象自身的内容（如字符串的replace()）。相反，这些方法会创建新的对象并返回，这样，就保证了不可变对象本身永远是不可变的。

在编写程序时，如果可以设计一个不变对象，那就尽量设计成不变对象
```

## 其他
1. 模块搜索路径：sys.path，在运行时修改模块搜索路径，运行结束后失效：sys.path.append('your_path')
2. 模块或函数有\_\_doc\_\_属性，可以直接访问，类似的还有\_\_name\_\_，\_\_author\_\_

# 数据类型

## 整型
1. 数值类型可以使用 "_" 分割，如 a = 10_000_000_000

## 浮点数
1. 浮点数的运算可能会有误差（4.2+2.1），
因为浮点数要转化为二进制进行计算，转化成的二进制小数可能是无限小数，只能截取部分存储与计算
2. 浮点数的有效数字为17位
3. 可以使用科学计数法表示

## 字符串
1. 多行字符串用'''或"""表示
2. python字符串是以Unicode编码的
3. ord() --> 获取单个字符的10进制整数编码，chr() --> 将编码转换成字符
4. replace('old', 'new') --> 不改变原始的，返回一个copy

## 布尔
1. 优先级：not > and > or

# 数据容器

## [数据容器常用操作的时间复杂度](https://wiki.python.org/moin/TimeComplexity)

## list[]
1. 增: append(value)/extend(list)/insert(index, value)
2. 删：pop()/pop(index)/remove(value)
3. 查：list[index]/list.index(value)/value in list --> O(n)
4. 改：list[index] = value
5. 排序：list.sort() --> INPLACE/sorted(list) --> 创建新的list

## tuple()
1. 增：不支持，追加数据是只能使用 "+" --> 创建新的tuple
2. 删：不支持
3. 查：tuple[index]/tuple.index(value)
4. 改：不支持
5. 排序：不支持，sorted(tuple) --> 创建排序后的list

```
* 一旦初始化不能更改，此不变指的是tuple的每个元素的指向不变，但指向的对象可能是可变的
```

## dict{}
1. 增：dict[key] = value
2. 删：dict.pop(key)
3. 查：dict[key]/key in dict --> O(1)/dict.get(key, default)
4. 改：dict[key] = value

```
* dict根据key来计算value的存储位置(Hash)，因此key必须是不可变的且可hash的
* 适用于高速查找的场景
```

## set{}
1. 增：set.add(key)
2. 删：set.remove(key)/set.pop() --> 不接受参数
3. 查：不支持
4. 改：不支持
5. 交：&
6. 并：|

```
* 无序和无重复元素(不可index)
* set根据key来计算key的存储位置(Hash)，由此来达到去重的目的
```

# 函数

## 函数基础
1. 函数名本身也是变量，其指向函数对象
2. 函数对象有一个**\_\_name\_\_**属性，可以拿到函数的名字

### return语句
1. retuen 和 return None 等价
2. return 多个值时其实返回的是一个tuple
### 函数的传入参数
#### 默认参数
默认参数在必须指向不变对象（反例：默认参数L为[]时，因为默认参数L也是一个变量，它指向对象[]，每次调用该函数，如果改变了L的内容，则下次调用时，默认参数的内容就变了，不再是函数定义时的[]了）
#### 可变参数
* params把接收到的参数存成一个tuple
```python
def func(*params):
    pass
## 调用
func(1,2,3,4)
func(*[1,2,3,4])
func(*(1,2,3,4))
```
#### 关键字参数
* 当函数中的有些参数是可选的时，可以使用关键字参数
* func()中，kw是关键字参数，关键字参数允许传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict（调用函数时没有kw，则kw={}）
* 缺点：函数的调用者可以传入任意不受限制的关键字参数。至于到底传入了哪些，需要在函数内部通过kw检查。
```python
def func(param1, param2, **kw):
    pass
## 调用
func(1, 2, city='Tianjin', univ='NanKai')
func(1, 2, **{'city':'Tianjin', 'univ':'NanKai'})
```
#### 命名关键字参数
* 命名关键字参数需要一个特殊分隔符*，*后面的参数被视为命名关键字参数
```python
def func(param1, param2, *, city, job):
    pass
```

* 函数定义中已经有了一个可变参数，后面跟着的命名关键字参数就不再需要一个特殊分隔符*了
* 和位置参数不同，命名关键字参数必须传入参数名。
```python
def func(param1, param2, *args, city='Tianjin', job):
    pass
## 调用
func(1,2,3,4, city='Tianjin', job='stu')
func(1,2,*[3,4], city='Tianjin', job='stu')
func(1,2,*[3,4], **{city='Tianjin', job='stu'})
func(1,2,*[3,4], **{job='stu'})
```
#### 参数定义的顺序
参数定义的顺序必须是：必选参数、默认参数、可变参数、命名关键字参数和关键字参数。

## 递归函数
递归函数：函数在内部调用自身

### 普通递归（计算阶乘）
在计算机中，递归通过栈来实现，每经过一次函数调用，栈会增加一层，函数 return 后栈减少一层，普通递归在 return 前需要计算 n*fact(n-1), 即需要再调用一次fact(),会导致栈的不断增长
```python
## 普通递归
def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n-1)
```

### 尾递归（计算阶乘）
尾递归 --> 函数返回的时候，调用自身本身，并且 return 语句“不能包含表达式”（表达式会导致return前调用函数进行计算），

```python
## 尾递归
def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)
## 调用
fact_iter(num, 1)
```

## 高阶函数（Higher-order function）
变量可以指向函数，函数的参数能接收变量，那么一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数。  
e.g.
```python
def add(x, y, f):
    return f(x) + f(y)
```

### map/reduce
#### map
**map(func, Iterable)**,map将传入的函数(func)依次作用到可迭代对象(Iterable)的每个元素，并把结果作为新的**Iterator**返回。

e.g.
```python
>>> list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
['1', '2', '3', '4', '5', '6', '7', '8', '9']
```
#### reduce
**reduce(func, Iterable)** 把一个函数作用在一个可迭代对象上，这个函数必须接收两个参数，reduce把结果继续和序列的下一个元素做累积计算，返回计算结束的值。
* reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)

eg1. 累计求和
```python
from functools import reduce
def add(x, y):
    return x + y

>>> reduce(add, [1, 3, 5, 7, 9])
25
```

eg2.
```python
from functools import reduce

def fn(x, y):
    return x * 10 + y

>>> reduce(fn, [1, 3, 5, 7, 9])
13579
```

### filter
**filter(func, Iterable)** 用于过滤序列,把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素,把结果作为新的**Iterator**返回。

eg. 删除 dict 中 value 为无效值的 item
```python
from functools import reduce

def func(dict_item):
    return dict_item[1]

>>> dict(filter(func, {'1':1, '2':0, '3':None, '4':4}.items()))
{'1': 1, '4': 4}
```

### sorted
sorted()函数也是一个高阶函数，它还可以接收一个key函数来实现自定义的排序，具体是：key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序

eg1. 按照绝对值大小排序
```python
>>> sorted([36, 5, -12, 9, -21], key=abs)
[5, 9, -12, -21, 36]
```

eg2. 忽略大小写，按字母顺序排序
```python
>>> sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=False)
['about', 'bob', 'Credit', 'Zoo']
```

eg3. 按value的大小排序
```python
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
>>> sorted(L, key=lambda x: x[1], reverse=False)
[('Bart', 66), ('Bob', 75), ('Lisa', 88), ('Adam', 92)]
```

## 返回函数

### eg1. 
可变参数求和,不需要立刻求和，而是在后面的代码中，根据需要再计算，其不返回求和的结果，而是返回求和的函数
```python
def lazy_sum(*args):
    def sum():
        ax = 0
        for n in args:
            ax = ax + n
        return ax
    return sum

>>> f = lazy_sum(1, 3, 5, 7, 9)
<function __main__.lazy_sum.<locals>.sum()>
>>> f()
25
```
此例中，内部函数sum可以引用外部函数lazy_sum的参数和局部变量，当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，这种程序结构称为 **“闭包（Closure）”**  

**当我们调用lazy_sum()时，每次调用都会返回一个新的函数，即使传入相同的参数:**
```python
>>> f1 = lazy_sum(1, 3, 5, 7, 9)
>>> f2 = lazy_sum(1, 3, 5, 7, 9)
>>> f1==f2
False
```

### eg2.
返回函数引用上层函数的循环变量
```python
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs

f1, f2, f3 = count()

>>> f1()
9
>>> f2()
9
>>> f3()
9
```
eg2. 中f1(), f2(), f3()全部都是9！原因就在于返回的函数引用了变量i，但它并非立刻执行。等到3个函数都返回时，它们所引用的变量i已经变成了3，因此最终结果为9。  

**返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。**  

如果返回函数一定要引用循环变量，方法是再创建一个函数，用该函数的参数绑定循环变量当前的值，无论该循环变量后续如何更改，已绑定到函数参数的值不变：
```python
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs = []
    for i in range(1, 4):
        fs.append(f(i)) # f(i)立刻被执行，因此i的当前值被传入f()
    return fs

f1, f2, f3 = count()

>>> f1()
1
>>> f2()
4
>>> f3()
9
```

## 匿名函数（lambda表达式）

### eg
**lambda x: x * x** 等价于以下函数：
```python
def f(x):
    return x * x
```
### 优缺点
优点：函数没有名字，不必担心函数名冲突。此外，匿名函数也是一个函数对象，也可以把匿名函数赋值给一个变量，再利用变量来调用该函数。  
缺点：只能有一个表达式，不用写return，返回值就是该表达式的结果。
### 作为返回值
```python
def build(x, y):
    return lambda: x * x + y * y
```

## 装饰器(Decorator)
不修改原函数体，在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator），**Decorator是一个返回函数的高阶函数**。

以下定义一个打印日志的装饰器log，其接受一个函数作为参数，并返回一个函数  
* 通过@语法将装饰器log至于函数now()的定义处，此时调用now()将执行在log()函数中返回的wrapper()函数。
* wrapper()函数的参数定义是(\*args, \*\*kw)，因此，wrapper()函数可以接受任意参数的调用。在wrapper()函数内，首先打印日志，再紧接着调用原始函数。

```python
import time

def log(func):
    @functools.wraps(func) # 把原始函数的__name__等属性复制到wrapper()函数中
    def wrapper(*args, **kw):
        print('\n@call %s()' % func.__name__)
        start = time.time()
        res = func(*args, **kw)
        print('runtime: %f seconds.' % (time.time() - start))
        return res
    return wrapper

@log
def now():
    print('2015-3-25')

>>> now()

@call now()
2015-3-25
runtime: 0.000020 seconds.
```

## 偏函数

functools.partial的作用就是，把一个函数的某些参数给固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。  
functools.partial(func, \*args, \*\*kw)创建偏函数时，接收函数对象、\*args和\*\*kw这3个参数

* eg1: int2将base默认值设置为2  
int2 = functools.partial(int, base=2)相当于：kw = { 'base': 2 }; int('10010', \*\*kw)
```python
>>> import functools
>>> int2 = functools.partial(int, base=2)
>>> int2('1000000')
64
```

* eg2: 
```python
max2 = functools.partial(max, 10)
#实际上会把10作为*args的一部分自动加到左边，也就是：
max2(5, 6, 7)
#相当于：
args = (10, 5, 6, 7)
max(*args)
```

# 高级特性

## 切片（适用于 list, tuple and string）
1. object[start:end:step]
2. 切片的start和end左闭右开
3. step = -1 返回一个逆序的对象
4. object[:] 返回一个对象的copy

## 列表生成式
1. for后面的if是一个筛选条件，不能带else  
[x for x in range(10) if x % 2 == 0]
2. 把if写在for前面必须加else,因为需要一个确定的返回值  
[x if x % 2 == 0 else 0 for x in range(10)]

## 生成器（generator）
**应用场景：** 创建一个包含100万个元素的列表，且列表元素可以按照某种算法推算

### 创建generate的方法

#### 推演规则简单时（用类似列表生成式来描述推演规则）
推演规则： x * x
```python
g = (x * x for x in range(10))
```
#### 推演规则复杂时（用函数来描述推演规则）
以斐波拉契数列为例 (1, 1, 2, 3, 5, 8, 13, 21, 34, ...)  
推演规则：除第一个和第二个数外，其他数等于前两个数的和
```python
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'
```

### 获取generator计算的值

#### next(generator_obj)

##### eg1
generator保存的是**算法**，每次调用next(g)，就计算出g的下一个元素的值，直到计算到最后一个元素，没有更多的元素时，抛出StopIteration的错误。

```python
g = (x * x for x in range(2))

print(next(g))
print(next(g))
print(next(g))

---

0
1
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-115-58d850acc428> in <module>
      3 print(next(g))
      4 print(next(g))
----> 5 print(next(g))

StopIteration: 
```

##### eg2
* generator和函数的执行流程不一样。函数是顺序执行，遇到return语句或者最后一行函数语句就返回。而变成generator的函数，在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。
* 下例在执行过程中，遇到yield就中断，下次又继续执行。执行3次yield后，已经没有yield可以执行了，所以，第4次调用next(o)就报错。

```python
def odd():
    print('step 1')
    yield 1
    print('step 2')
    yield(3)
    print('step 3')
    yield(5)
    return 'done'

o = odd()

print(next(o))
print(next(o))
print(next(o))
print(next(o))

---

step 1
1
step 2
3
step 3
5
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-121-5cfad42f1d43> in <module>
     12 print(next(o))
     13 print(next(o))
---> 14 print(next(o))

StopIteration: done
```

#### for loop

##### eg1
```python
g = (x * x for x in range(2))

##generator是可迭代对象
from collections import Iterable
print(isinstance(g, Iterable))

## 获取值
for n in g:
    print(n)
    
---
True
0
1
```

##### eg2
用 while n < max 控制不会一直循环获取 yield

```python
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

g = fib(5)

for n in g:
    print(n)
    
---
1
1
2
3
5
```

##### eg3
用for循环调用generator时，发现拿不到generator的return语句的返回值。如果想要拿到返回值，必须捕获StopIteration错误，返回值包含在StopIteration的value中

```python
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'

g = fib(5)

while True:
    try:
        x = next(g)
        print('g:', x)
    except StopIteration as e:
        print('Generator return value:', e.value)
        break

---
g: 1
g: 1
g: 2
g: 3
g: 5
Generator return value: done
```

## 迭代器（Iterator）

### 可迭代对象（Iterable）
* list, tuple, dict, set, str, generator  
* 判断一个对象是否是**可迭代对象**：
```python
from collections.abc import Iterable
isinstance({}, Iterable)
```

### 迭代器定义
* 可以被next()函数调用并不断返回下一个值的对象称为迭代器：Iterator  
* 生成器都是Iterator对象，但list、dict、str虽然是Iterable，却不是Iterator  
* 把list、dict、str等Iterable变成Iterator可以使用iter()函数

判断一个对象是否是Iterator
```python
from collections.abc import Iterator
isinstance((x for x in range(10)), Iterator)

isinstance(iter([]), Iterator)
```
### 迭代器的作用
Python的Iterator对象表示的是一个数据流，Iterator对象可以被next()函数调用并不断返回下一个数据，直到没有数据时抛出StopIteration错误。可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过next()函数实现按需计算下一个数据。

Iterator可以表示一个无限大的数据流，例如全体自然数。而使用list永远不可能存储全体自然数。

### 关于for循环的实现
Python的for循环本质上是通过不断调用next()函数实现的：

```python
for x in [1, 2, 3, 4, 5]:
    print(x)

##等价于

it = iter([1, 2, 3, 4, 5])
while True:
    try:
        x = next(it)
        print(x)
    except:
        break
```

# 面向对象编程（OOP）

## 类与实例（对象）基础

* 和静态语言不同，Python允许对实例变量绑定任何数据  
eg. 给实例绑定方法
```python
>>> def set_age(self, age): # 定义一个函数作为实例方法
...     self.age = age
...
>>> from types import MethodType
>>> s.set_age = MethodType(set_age, s) # 给实例绑定一个方法
>>> s.set_age(25) # 调用实例方法
>>> s.age # 测试结果
25
```
* 类中方法的第一个参数self指向创建的实例本身

### 创建类
e.g.
```python
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
s = Student(name=1, score=2)
s.extra_data = 1 #对实例变量绑定任何__init__之外的其他数据
```
### 数据封装和访问限制
* 数据封装是指在类的内部定义数据的访问方法  
* 实例的变量名如果以\_\_开头，就变成了一个私有变量（private），只有内部可以访问，外部实例不能直接访问。  
* 访问限制确保了外部代码不能随意修改对象内部的状态，这样通过访问限制的保护，代码更加健壮。

### 继承与多态
* 子类对象可以直接访问父类的方法
* 多态：子类和父类方法重名时覆盖父类的方法

### 获取对象信息
1. 优先使用isinstance()判断对象的类型，判断指定类型及其父类时均为True。  
2. 使用dir()函数获得一个对象的所有属性和方法，它返回一个包含字符串的list  

e.g. 获得字符串对象的属性和方法  
```python
>>> dir('ABC')
['__add__', '__class__',..., '__subclasshook__', 'capitalize', 'casefold',..., 'zfill']
```

3. 内置函数 getattr()、setattr()以及hasattr()用于获得、设置、判断对象的属性或方法

e.g.
```python
>>> setattr(obj, 'y', 19) # 设置一个属性'y'
>>> hasattr(obj, 'y') # 有属性'y'吗？
>>> fn = getattr(obj, 'power') # 获取属性'power'并赋值到变量fn
```

### 实例属性和类属性
Python是动态语言，根据类创建的实例可以任意绑定属性，但是在编写程序的时候，不要对实例属性和类属性使用相同的名字，因为相同名称的实例属性将屏蔽掉类属性，当删除实例属性后（del obj.attr），再使用相同的名称，访问到的将是类属性。

## 面向对象编程高级编程




### \_\_slots\_\_
Python允许在定义class的时候，定义一个特殊的__slots__变量，来限制该class实例能添加的属性  
```python
class Student(object):
    __slots__ = ('name', 'age') # 用tuple定义允许绑定的属性名称
```

使用\_\_slots\_\_要注意，\_\_slots\_\_定义的属性仅对当前类实例起作用，对继承的子类不起作用，除非在子类中也定义\_\_slots\_\_，这样，子类实例允许定义的属性就是自身的\_\_slots\_\_加上父类的\_\_slots\_\_

### @property
Python内置的@property装饰器就是负责把一个方法变成属性调用
```python
class Student(object):
    
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
        
    @property
    def age(self):
        return 2015 - self._birth
    
--------------------------------------------------------    
>>> s = Student()
>>> s.score = 60 # OK，实际转化为s.set_score(60)
>>> s.score # OK，实际转化为s.get_score()
60
>>> s.score = 9999
Traceback (most recent call last):
  
ValueError: score must between 0 ~ 100!
```
**其中age是只读属性**

### 多重继承










# 递归
1. 处理递归时：不要跳进递归，而是利用明确的定义来实现算法逻辑。


```python
def fact(n):
    print(f'传入的参数n: {n}')
    if n==1:
        print('递归完成\n')
        n-=1
        return None
    
    fact(n - 1)
    fact(n - 1)
    
    print(f'n**2: {n**2}')

fact(3)
```

    传入的参数n: 3
    传入的参数n: 2
    传入的参数n: 1
    递归完成
    
    传入的参数n: 1
    递归完成
    
    n**2: 4
    传入的参数n: 2
    传入的参数n: 1
    递归完成
    
    传入的参数n: 1
    递归完成
    
    n**2: 4
    n**2: 9


## [反转链表](https://labuladong.gitbook.io/algo/shu-ju-jie-gou-xi-lie/shou-ba-shou-shua-lian-biao-ti-mu-xun-lian-di-gui-si-wei/di-gui-fan-zhuan-lian-biao-de-yi-bu-fen)

### 常规反转链表


```python
## 迭代
def reverse(head):
    pre = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt

    return pre

## 递归
def reverse(head):
    if not head:
        return None
    ## 递归结束条件（结束时返回反转部分的头节点,头节点指向None）,只有一个节点时返回自己
    if head.next is None:
        return head
    last = reverse(head.next) ## 完成节点[2:n]的反转,并返回反转后的头节点（最后一个节点）,此时node1仍指向node2,node2指向None
    head.next.next = last
    head.next = None
    return last
```

### 反转链表的前n个节点


```python
def reverseN(head, n):
    if not head or n == 0:
        return head
    if n < 0:
        return None
    
    ## 递归结束条件
    if n == 1:
        tail = head.next
        return head
    
    last = reverseN(head.next, n - 1)
    head.next.next = last
    head.next = tail
    return last
```

### 反转区间[m, n]之间的链表元素


```python
def reverseMN(head, m, n):
    if m == 1:
        return reverseN(head, n)
    head.next = reverseMN(head.next, m-1, n-1)
    return head
```

### k个一组反转链表(迭代+递归)


```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
    
        a = head
        b = head
        for _ in range(k):
            if not b:
                return head
            b = b.next
        
        ## 反转节点[a, b)之间的节点, 不包括b节点
        newHead = self.reverseBetweenAB(a,b)
        
        ##递归反转链表并进行连接
        a.next = self.reverseKGroup(b,k)
        return newHead

    def reverseBetweenAB(self, a, b):
        pre = None
        cur = a
        while cur != b:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre
```

## 链表的前序和后续遍历


```python
def preorder(head):
    if not head:
        return None
    print(head.val)
    preorder(head.next)

def postorder(head):
    if not head:
        return None
    postorder(head.next)
    print(head.val)
```

### 判断链表是否是回文链表（正序=逆序）
1. 方法一：反转链表，和正序链表一一比较
2. 方法二：存储成list，比较list==list[::-1]  
~~3. 方法三：通过后序遍历比较，如下~~


```python
def iscorrect(head):
    def traverse(right):
        if right is None:
            return True
        ## 后续遍历
        rst = traverse(right.next)
        rst = rst and right.val == left.val
        left = left.next
        return rst
    
    
    left = head
    return traverse(head)
    
    
            
```

