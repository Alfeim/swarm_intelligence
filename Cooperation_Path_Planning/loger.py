DivideLineLength = 20
class Logger:
    """The logger class
    
    if the global parm OpenLogger is True,then the logger
    istance will write to files
    
    Attributes:
        file: the log file
    """

    def __init__(self, output_file='./Log.txt'):
        self.file = None
        try:
            self.file = open(output_file, 'w+')
        except:
            print("can not open log file")


    def __del__(self):
        try:
            self.file.close()
        except:
            return

    def log(self, msg):
        if(self.file is not None):
            self.file.writelines(msg + '\n')

    def logDivideLine(self, msg, length=DivideLineLength, head=True):
        if(self.file is None):
            return
        while(length < len(msg)):
            length += 20

        star_counts = length - len(msg)
        left_counts = star_counts // 2
        right_counts = star_counts - left_counts
        self.file.writelines('\n'*(1 if head else 0) +'*'*left_counts + msg + '*'*right_counts + '\n')

