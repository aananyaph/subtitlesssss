"""Module 1 - List Windows audio devices, highlighting WASAPI loopback-capable ones."""
import pyaudiowpatch as pyaudio


def main():
    p = pyaudio.PyAudio()

    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("WASAPI host API not available on this system.")
        p.terminate()
        return

    print(f"Default WASAPI host: {wasapi_info.get('name')}")
    print(f"Default output device: {p.get_device_info_by_index(wasapi_info['defaultOutputDevice']).get('name')}")
    print()

    print("=== Output devices (WASAPI) ===")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("hostApi") != wasapi_info["index"]:
            continue
        if info.get("maxOutputChannels") == 0:
            continue
        is_loopback = info.get("isLoopbackDevice")
        marker = "  <-- LOOPBACK" if is_loopback else ""
        print(f"[{i}] {info.get('name')} | {info.get('maxOutputChannels')}ch | {info.get('defaultSampleRate')} Hz{marker}")

    print()
    print("=== All WASAPI loopback devices ===")
    found = False
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("hostApi") == wasapi_info["index"] and info.get("isLoopbackDevice"):
            found = True
            print(f"[{i}] {info.get('name')} | {info.get('maxInputChannels')}ch | {info.get('defaultSampleRate')} Hz")

    if not found:
        print("No loopback devices found. Make sure something is playing audio.")

    p.terminate()


if __name__ == "__main__":
    main()
