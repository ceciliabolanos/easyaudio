import numpy as np

def am_tone(f_carrier, f_modulation, modulation_index, message_phase=0, init_phase=0, duration=1000, fs=16000):
    t = np.arange(int(fs*duration/1000))/fs
    return (1+modulation_index*np.cos(2*np.pi*f_modulation*t + message_phase))*np.sin(2*np.pi*f_carrier*t + init_phase)

def fm_tone(f_carrier, f_modulation, f_deviation, amplitude=1, init_phase=0, duration=1000, fs=16000):
    t = np.arange(int(fs*duration/1000))/fs
    return amplitude*np.cos(init_phase + 2*np.pi*f_carrier*t + (f_deviation/f_modulation)*np.sin(2*np.pi*f_modulation*t))

def pure_tone(f, amplitude=1, init_phase=0, duration=1000, fs=16000):
    t = np.arange(int(fs*duration/1000))/fs
    return amplitude*np.sin(2*np.pi*f*t + init_phase)