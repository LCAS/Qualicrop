import ctypes
import os
import numpy as np
import time
import logger

def object_to_voidp(obj):
    return ctypes.cast(ctypes.pointer(ctypes.py_object(obj)), ctypes.c_void_p)
    
def voidp_to_object(voidp):
    return ctypes.cast(voidp, ctypes.POINTER(ctypes.py_object)).contents.value
    
def prestartCallback(handle,feature,context):
    ss = voidp_to_object(context)
        
    [height, err] = ss.getint('Camera.Image.Height')
    [width, err] = ss.getint('Camera.Image.Width')

    ss.width = width
    ss.height = height

    return 0
    
class Sensor:
    def __init__(self, dll, index, sspName, handle):
        self.dll = dll
        self.opened = True
        self.logger = logger.logger()
        
        self.index = index
        self.handle = handle
        self.sspname = sspName
        
        self.width = None
        self.height = None
        
        self.featurecallbacks = {}
        self.featurecallbackcontexts = {}

        self.datacallback = None
        self.datacallbackcontext = None
    
        self.registerInternalCallbacks()
        
    def open(self):
        self.logger.debug('Sensor::open called')
        
        if self.isopen() == True:
            return 0
        
        error = self.dll.SI_Open(self.index, ctypes.byref(self.handle))
        
        if error == 0:
            self.opened = True
            self.registerInternalCallbacks()
                
        return error
        
    def reopen(self):
        self.logger.debug('Sensor::reopen')
        self.close()
        return self.open()
        
    def close(self):
        self.logger.debug('Sensor::close')
        
        if self.isopen() == False:
            return 0

        # Sleep for awhile, just because...
        time.sleep(0.5)

        # First always stop acquiring images.
        self.command('Acquisition.Stop')

        # Then unregister all internal callbacks
        self.unregisterInternalCallbacks()

        # Finally close the sensor.
        error = self.dll.SI_Close(self.handle)

        if error == 0:
            self.opened = False

        return error
    
    def reinit(self):
        self.logger.debug('Sensor::reinit')
        
        if self.isopen() == True:
            self.close()
        
        openError = self.open()
        
        if openError != 0:
            return openError
            
        return self.command('Initialize')
        
    def isopen(self):
        self.logger.debug('Sensor::isopen')
        return self.opened
    
    def isinitialized(self):
        self.logger.debug('Sensor::isinitialized')
        isOpen = self.isopen()
                
        if isOpen == False:
            return False
        
        (isInit, err) = self.getbool('IsInitialized')
        
        return isInit == True and err == 0
    
    def registerInternalCallbacks(self):
        self.logger.debug('Sensor::registerInternalCallbacks')
        # Add internal callback registration here.
        ret = self.registerfeaturecallback('Acquisition.PreStartCallback', prestartCallback, self)
        
    def unregisterInternalCallbacks(self):
        self.logger.debug('Sensor::unregisterInternalCallbacks')
        # Always remember to unregister them as well!!!
        self.unregisterfeaturecallback('Acquisition.PreStartCallback')
        
    def command(self, handle,feature):
        self.logger.debug('Sensor::command {}'.format(feature))
        #return self.dll.SI_Command(self.handle, feature)
        return self.dll.SI_Command(handle, feature)

    def isimplemented(self, feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsImplemented(self.handle, feature, ctypes.byref(valuebool))
        self.logger.debug('Sensor::isimplemented {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def isreadonly(self,feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsReadOnly(self.handle, feature, ctypes.byref(valuebool))
        self.logger.debug('Sensor::isreadonly {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def iswritable(self,feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsWritable(self.handle, feature, ctypes.byref(valuebool))
        self.logger.debug('Sensor::iswritable {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def isreadable(self,feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsReadable(self.handle, feature, ctypes.byref(valuebool))
        self.logger.debug('Sensor::isreadable {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)
    
    def setfloat(self, feature, value):
        self.logger.debug('Sensor::setfloat {} <- {}'.format(feature, value))
        valueDouble = ctypes.c_double(value)
        return self.dll.SI_SetFloat(self.handle, feature, valueDouble)

    def getfloat(self, feature):
        valueDouble = ctypes.c_double()
        error = self.dll.SI_GetFloat(self.handle, feature, ctypes.byref(valueDouble))
        self.logger.debug('Sensor::getfloat {} -> {}'.format(feature, valueDouble.value))
        return (valueDouble.value, error)

    def setint(self, feature, value):
        self.logger.debug('Sensor::setint {} <- {}'.format(feature, value))
        valueint = ctypes.c_int64(value)
        return self.dll.SI_SetInt(self.handle, feature, valueint)

    def getint(self, feature):
        valueint = ctypes.c_int64()
        error = self.dll.SI_GetInt(self.handle, feature, ctypes.byref(valueint))
        self.logger.debug('Sensor::getint {} -> {}'.format(feature, valueint.value))
        return (valueint.value, error)

    def getintmax(self, feature):
        valueint = ctypes.c_int64()
        error = self.dll.SI_GetIntMax(self.handle, feature, ctypes.byref(valueint))
        self.logger.debug('Sensor::getintmax {} -> {}'.format(feature, valueint.value))
        return (valueint.value, error)

    def getintmin(self, feature):
        valueint = ctypes.c_int64()
        error = self.dll.SI_GetIntMin(self.handle, feature, ctypes.byref(valueint))
        self.logger.debug('Sensor::getintmin {} -> {}'.format(feature, valueint.value))
        return (valueint.value, error)
    
    def getfloatmax(self, feature):
        valueDouble = ctypes.c_double()
        error = self.dll.SI_GetFloatMax(self.handle, feature, ctypes.byref(valueDouble))
        self.logger.debug('Sensor::getfloatmax {} -> {}'.format(feature, valueDouble.value))
        return (valueDouble.value, error)

    def getfloatmin(self, feature):
        valueDouble=ctypes.c_double()
        error=self.dll.SI_GetFloatMin(self.handle,feature,ctypes.byref(valueDouble))
        self.logger.debug('Sensor::getfloatmin {} -> {}'.format(feature, valueDouble.value))
        return (valueDouble.value,error)

    def setbool(self, feature, value):
        self.logger.debug('Sensor::setbool {} <- {}'.format(feature, value))
        valuebool = ctypes.c_bool(value)
        return self.dll.SI_SetBool(self.handle, feature, valuebool)

    def getbool(self, feature):
        valuebool = ctypes.c_bool()
        error = self.dll.SI_GetBool(self.handle, feature, ctypes.byref(valuebool))
        self.logger.debug('Sensor::getbool {} -> {}'.format(feature, valuebool.value))
        return (valuebool.value, error)

    def setstring(self, feature, str_value):
        self.logger.debug('Sensor::setstring {} <- {}'.format(feature,str_value))
        return self.dll.SI_SetString(self.handle, feature, str_value)
           
    def getstring(self, feature):
        maxSize = 255
        valueStringPtr = ctypes.create_unicode_buffer(maxSize)
        error = self.dll.SI_GetString(self.handle, feature, valueStringPtr, maxSize)
        self.logger.debug('Sensor::getstring {} -> {}'.format(feature, valueStringPtr.value))
        return (valueStringPtr.value, error)
    
    def getstringmaxlength(self, feature):
        value = ctypes.c_int()
        error = self.dll.SI_GetStringMaxLength(self.handle, feature, ctypes.byref(value))
        self.logger.debug('Sensor::getstringmaxlength {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def getenumstringmaxlength(self, feature, index):
        index = ctypes.c_int(index)
        value = ctypes.c_int()
        error = self.dll.SI_GetStringMaxLength(self.handle, feature, index, ctypes.byref(value))
        self.logger.debug('Sensor::getenumstringmaxlength {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def setenumindexbystring(self, feature, str_val):
        self.logger.debug('Sensor::setenumindexbystring {} <- {}'.format(feature, str_val))
        return self.dll.SI_SetEnumIndexByString(self.handle, feature, str_val)

    def setenumindex(self,feature,index):
        self.logger.debug('Sensor::setenumindex {} <- {}'.format(feature, index))
        return self.dll.SI_SetEnumIndex(self.handle,feature,index)

    def setenumstring(self, feature, str_val):
        self.logger.debug('Sensor::setenumstring {} <- {}'.format(feature, str_val))
        return self.dll.SI_SetEnumString(self.handle, feature, str_val)

    def getenumindex(self, feature):
        value = ctypes.c_int()
        error = self.dll.SI_GetEnumIndex(self.handle, feature, ctypes.byref(value))
        self.logger.debug('Sensor::getenumindex {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def getenumcount(self, feature):
        value = ctypes.c_int()
        error = self.dll.SI_GetEnumCount(self.handle, feature, ctypes.byref(value))
        self.logger.debug('Sensor::getenumcount {} -> {}'.format(feature, value.value))
        return (value.value, error)

    def getenumstringbyindex(self,feature, index):
        index = ctypes.c_int(index)
        maxSize = 255
        valueStringPtr = ctypes.create_unicode_buffer(maxSize)
        error = self.dll.SI_GetEnumStringByIndex(self.handle, feature, index, valueStringPtr,maxSize)
        self.logger.debug('Sensor::getenumstringbyindex {} -> {}'.format(feature, valueStringPtr.value))
        return (valueStringPtr.value, error)

    def getenum(self,feature):
        (count,error) = self.getenumcount(feature)
        enumList = []
        
        for i in range(count):
            (str, error) = self.getenumstringbyindex(feature,i)
            enumList.append(str)
        
        (enumIndex, error) = self.getenumindex(feature)

        self.logger.debug('Sensor::getenum {} -> {} i:{}'.format(feature, enumList, enumIndex))
        
        return (enumList, enumIndex, error)                       

    def isenumindexavailable(self, feature, index):
        index = ctypes.c_int(index)
        valuebool = ctypes.c_bool()
        error = self.dll.SI_IsEnumIndexAvailable(self.handle, feature, index, ctypes.byref(valuebool))
        self.logger.debug('Sensor::isenumindexavailable {} -> {}'.format(feature, index))
        return (valuebool.value, error)

    def registerfeaturecallback(self, feature, callbackFunct, context):
        self.logger.debug('Sensor::registerfeaturecallback {}'.format(feature))
        if self.checkCallbackExists(feature) == True:
            return -1
        
        featureCB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p)
        self.featurecallbacks[feature] = featureCB(callbackFunct)
        userdata = object_to_voidp(context)
        self.featurecallbackcontexts[feature] = userdata
        
        return self.dll.SI_RegisterFeatureCallback(self.handle, feature, self.featurecallbacks[feature], self.featurecallbackcontexts[feature])
        
    def unregisterfeaturecallback(self, feature):
        self.logger.debug('Sensor::unregisterfeaturecallback {}'.format(feature))
        if self.checkCallbackExists(feature) == False:
            return -1

        self.dll.SI_UnregisterFeatureCallback(self.handle, feature, self.featurecallbacks[feature])
        
        del self.featurecallbacks[feature]
        del self.featurecallbackcontexts[feature]
        
        return 0
        
    def checkCallbackExists(self, feature):
        return (feature in self.featurecallbacks)

    def registerDataCallback(self, callbackFunct):
        self.logger.debug('Sensor::registerDataCallback')
        #dataCB = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p)
        dataCB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p)
        self.datacallback = dataCB(callbackFunct)
        self.datacallbackcontext = ctypes.c_void_p(0)
        
        return self.dll.SI_RegisterDataCallback(self.handle, self.datacallback, self.datacallbackcontext)
    
    def unregisterDataCallback(self):
        return self.dll.SI_UnregisterDataCallback(self.handle)
        
    def wait(self, timeout, latest = False):
    
        if latest == True:
            self.command("Acquisition.RingBuffer.Sync")
        
        tout = ctypes.c_int64(timeout)
        framesize = ctypes.c_int64()
        framenumber = ctypes.c_int64()
        size = self.width * self.height * 2  
        frar = bytearray(size)
        type = ctypes.c_char * size
        frame = type.from_buffer(frar) 
        error = self.dll.SI_Wait(self.handle,frame,ctypes.byref(framesize),ctypes.byref(framenumber),timeout)
        data_ptr = ctypes.cast(frame, ctypes.POINTER(ctypes.c_uint16))
        array = np.ctypeslib.as_array(data_ptr, shape=(self.height, self.width))
        return (array, framenumber, error)
