# import os
# import sounddevice as sd
# from scipy.io.wavfile import write

# output_dir="captured_voices"
# os.makedirs(output_dir,exist_ok=True)
# sample_rate = 16000   
# duration = 2          
# persons = ["Krishnendu","Jithesh","Ajay","Unknown"]

# for person in persons:
#    for i in range(1,6):
#        print(f"\n Recording voice for {person}")
#        input("Press ENTER and speak")
#        audio = sd.rec(int(duration * sample_rate),
#                samplerate=sample_rate,
#                channels=1,
#                dtype='int16')
#        sd.wait()
#        filename = os.path.join(output_dir, f"{person}_voice_{i}.wav")

#        write(filename, sample_rate, audio)
#        print(f"Saved {filename}")




import os
import sounddevice as sd
from scipy.io.wavfile import write

output_dir = "captured_voices"
os.makedirs(output_dir, exist_ok=True)

sample_rate = 16000
duration = 2
person = "Jithesh"   

for i in range(1, 6):  
    print(f"\nRecording voice for {person}")
    input("Press ENTER and speak")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='int16')
    sd.wait()
    filename = os.path.join(output_dir, f"{person}_voice_{i}.wav")
    write(filename, sample_rate, audio)
    print(f"Saved {filename}")