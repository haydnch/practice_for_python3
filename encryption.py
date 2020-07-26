import hashlib


def md5(str = 'It is a test'):
	# 创建md5对象
	md5_str = hashlib.md5()

	# 此处必须声明encode
	# 若写法为hl.update(str)  报错为： Unicode-objects must be encoded before hashing
	md5_str.update(str.encode(encoding='utf-8'))
	print('MD5加密前为 ：' + str)
	print('MD5加密后为 ：' + md5_str.hexdigest())
	return md5_str.hexdigest()

if __name__ == '__main__':
	#str = input("请输入你要加密的字符串: ")
	md5()
