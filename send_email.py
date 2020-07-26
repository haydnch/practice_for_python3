import yagmail

yag_server = yagmail.SMTP(user = 'haydnch@foxmail.com', password ='your token',
                          host = 'smtp.qq.com')
email_to = ['leslie12956.ch98@foxmail.com',]
email_title = '测试报告'
email_content = 'I really love Lisongzhou\n'
email_attachments = ['username.txt',]

yag_server.send(email_to, email_title, email_content, email_attachments)

yag_server.close()
