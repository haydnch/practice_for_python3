import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt

def bubble_sort(needSortList):
    for i in range(len(needSortList) - 1):
        #设置一个标志以确认列表是否已经完成排序
        isSorted = True
        for j in range(len(needSortList) - 1 - i):
            if needSortList[j] > needSortList[j + 1]:
                needSortList[j], needSortList[j + 1] = needSortList[j + 1], needSortList[j]
                #这个时候排序还在进行说明排序未完成，程序应该继续执行下去
                isSorted = False
        #如果isSorted没有被设置为True，说明在某一轮排序时所有的元素位置都没有发生变化，这个时候排
        #序就已经完成了，我们需要让程序停止执行了。
        if isSorted:
            break

def selection_sort(myList):
    #获取list的长度
    length = len(myList)
    #一共进行多少轮比较
    for i in range(0,length-1):
        #默认设置最小值得index为当前值
        smallest = i
        #用当先最小index的值分别与后面的值进行比较,以便获取最小index
        for j in range(i+1,length):
            #如果找到比当前值小的index,则进行两值交换
            if myList[j]<myList[smallest]:
                tmp = myList[j]
                myList[j] = myList[smallest]
                myList[smallest]=tmp

def insertion_sort(arr):
    length = len(arr)
    for i in range(1,length):
        x = arr[i]
        for j in range(i,-1,-1):
            # j为当前位置，试探j-1位置
            if x < arr[j-1]:
                arr[j] = arr[j-1]
            else:
                # 位置确定为j
                break
        arr[j] = x

def shell_insert_sort(a,dk):
    n = len(a)
    for k in range(dk): # 间距取dk，一共可以组成dk个子序列
        for i in range(k+dk,n,dk):#第0，dk,2dk....为一组
            temp = a[i] # 记录待插入的元素值
            j = i - dk # 子序列的前一个元素
            while j>=k and a[j]>temp:# 寻找插入的位置
                a[j+dk] = a[j]
                j = j - dk
            a[j+dk] = temp # 插入


def shell_sort(a):
    n = len(a)
    dk = n//2 #取第一个dk，长度的一半
    while dk>=1:
        shell_insert_sort(a,dk)
        dk = dk//2


def merge_sort( li ):
    #不断递归调用自己一直到拆分成成单个元素的时候就返回这个元素，不再拆分了
    if len(li) == 1:
        return li

    #取拆分的中间位置
    mid = len(li) // 2
    #拆分过后左右两侧子串
    left = li[:mid]
    right = li[mid:]

    #对拆分过后的左右再拆分 一直到只有一个元素为止
    #最后一次递归时候ll和lr都会接到一个元素的列表
    # 最后一次递归之前的ll和rl会接收到排好序的子序列
    ll = merge_sort( left )
    rl =merge_sort( right )

    # 我们对返回的两个拆分结果进行排序后合并再返回正确顺序的子列表
    # 这里我们调用拎一个函数帮助我们按顺序合并ll和lr
    return merge(ll , rl)

#这里接收两个列表
def merge( left , right ):
    # 从两个有顺序的列表里边依次取数据比较后放入result
    # 每次我们分别拿出两个列表中最小的数比较，把较小的放入result
    result = []
    while len(left)>0 and len(right)>0 :
        #为了保持稳定性，当遇到相等的时候优先把左侧的数放进结果列表，因为left本来也是大数列中比较靠左的
        if left[0] <= right[0]:
            result.append( left.pop(0) )
        else:
            result.append( right.pop(0) )
    #while循环出来之后 说明其中一个数组没有数据了，我们把另一个数组添加到结果数组后面
    result += left
    result += right
    return result


def heap_sort(input_list):
    # 调整parent结点为大根堆
    def HeapAdjust(input_list, parent, length):

        temp = input_list[parent]
        child = 2 * parent + 1

        while child < length:
            if child + 1 < length and input_list[child] < input_list[child + 1]:
                child += 1

            if temp > input_list[child]:
                break
            input_list[parent] = input_list[child]
            parent = child
            child = 2 * child + 1
        input_list[parent] = temp

    if input_list == []:
        return []
    sorted_list = input_list
    length = len(sorted_list)
    # 最后一个结点的下标为length//2-1
    # 建立初始大根堆
    for i in range(0, length // 2)[::-1]:
        HeapAdjust(sorted_list, i, length)

    for j in range(1, length)[::-1]:
        # 把堆顶元素即第一大的元素与最后一个元素互换位置
        temp = sorted_list[j]
        sorted_list[j] = sorted_list[0]
        sorted_list[0] = temp
        # 换完位置之后将剩余的元素重新调整成大根堆
        HeapAdjust(sorted_list, 0, j)
    return sorted_list


def quick_sort(a, left, right):
    if (left < right):
        mid = partition(a, left, right)
        quick_sort(a, left, mid - 1)
        quick_sort(a, mid + 1, right)


def partition(a, left, right):
    x = a[right]
    i = left - 1  # 初始i指向一个空，保证0到i都小于等于 x
    for j in range(left, right):  # j用来寻找比x小的，找到就和i+1交换，保证i之前的都小于等于x
        if (a[j] <= x):
            i = i + 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[right] = a[right], a[i + 1]  # 0到i 都小于等于x ,所以x的最终位置就是i+1
    return i + 1


if __name__ == '__main__':  # 生成n个0-10万的随机整型数据
    n = 10000  # n in [10, 100, 1000, 10000, 100000]
    arr = [random.randint(0, n) for i in range(n)]
    cost_time = {}
    start1 = time.time()  # 使用deepcopy是为了排除电脑或程序自动优化或使用缓存等因素
    bubble_sort(deepcopy(arr))
    print("冒泡排序耗时：" + str(time.time() - start1))
    cost_time['冒泡排序']=time.time() - start1

    start2 = time.time()
    selection_sort(deepcopy(arr))
    print("选择排序耗时：" + str(time.time() - start2))
    cost_time['选择排序'] = time.time() - start2

    start3 = time.time()
    insertion_sort(deepcopy(arr))
    print("插入排序耗时：" + str(time.time() - start3))
    cost_time['插入排序'] = time.time() - start3

    start4 = time.time()
    shell_sort(deepcopy(arr))
    print("希尔排序耗时：" + str(time.time() - start4))
    cost_time['希尔排序'] = time.time() - start4

    start5 = time.time()
    merge_sort(deepcopy(arr))
    print("归并排序耗时：" + str(time.time() - start5))
    cost_time['归并排序'] = time.time() - start5

    start6 = time.time()
    heap_sort(deepcopy(arr))
    print("堆排序耗时：" + str(time.time() - start6))
    cost_time['堆排序'] = time.time() - start6

    start7 = time.time()
    quick_sort(deepcopy(arr),0,len(arr)-1)
    print("快速排序耗时：" + str(time.time()-start7))
    cost_time['快速排序'] = time.time() - start7

    x_data = list(cost_time.keys())
    y_data = list(cost_time.values())

    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文",
    plt.rcParams['axes.unicode_minus'] = False       #显示负号"
    plt.figure(figsize=(6,4), dpi=300)
    plt.plot(x_data, y_data, 'o-', color='#4169E1', label='排序耗时')
    for x, y in zip(x_data, y_data):
        plt.text(x, y + 0.3, '%.3f' % y, ha='center', va='bottom', fontsize=10.5)
    plt.legend(loc=1)
    plt.ylabel('排序时间/s')
    plt.title('对10000个Integer进行排序', fontsize = 15)
    plt.savefig('test_sort.png')
    plt.show()