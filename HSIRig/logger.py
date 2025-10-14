import logging
import os
import sys

g_filePath = 'c:/temp/log.txt'

def setFilepath(filepath):
    global g_filePath
    g_filePath = filepath
    
class logger:
    class __logger:
        def __init__(self):
            global g_filePath
            self.logger = logging.getLogger(g_filePath)
        
            if self.logger.hasHandlers() == False:
                dirPath = os.path.dirname(g_filePath)            
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)
                    
                self.fileHandler = logging.FileHandler(g_filePath)
                self.streamHandler = logging.StreamHandler(sys.stdout)
                self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                self.fileHandler.setFormatter(self.formatter)
                self.logger.addHandler(self.fileHandler)
                self.logger.addHandler(self.streamHandler)
                self.logger.setLevel(logging.DEBUG)
            else:
                self.fileHandler = self.logger.handlers[0]
                self.streamHandler = self.logger.handlers[1]
                
        def debug(self, message):
            self.logger.info(message)
            self.fileHandler.flush()
            self.streamHandler.flush()
            
        def warning(self, message):
            self.logger.warning(message)
            self.fileHandler.flush()
            self.streamHandler.flush()
            
        def error(self, message):
            self.logger.error(message)
            self.fileHandler.flush()
            self.streamHandler.flush()
            
    instance = None
    
    def __init__(self):
        if not logger.instance:
            logger.instance = logger.__logger()
            
    def __getattr__(self, name):
        return getattr(self.instance, name)
