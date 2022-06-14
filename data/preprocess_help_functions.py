import sys

def detect_language_error(serie):
    '''Function to understand which rows generate errors when detecting the language of a message '''
    for i in range(len(serie)):
        try: 
            lang = detect(serie[i])
            #df['lang'][i]=lang
        except:                                                       
            lang ='error'                                                  
            print("This row throws error:", {i}) 



def translate_function(serie):
    '''Function to translate - not working'''
    translator = Translator()
    #serie2 = serie2.astype(str)
    serie = translator.translate(serie,dest = 'en')
    return serie



DL_Local_HR_Services_BG@sap.com