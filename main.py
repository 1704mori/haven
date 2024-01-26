from bot import Bot
import mss
import threading
import cv2
import numpy as np

def capture_and_process(bot):
    while True:
        screenshot = mss.mss()
        screen = screenshot.grab({
            "top": 0,
            "left": 0,
            "width": 1920,
            "height": 1080,
        })

        frame = np.array(screen)
        bot.process_image(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    bot_instance = Bot()
    bot_instance.debug(True)
    bot_thread = threading.Thread(target=capture_and_process, args=(bot_instance,))
    bot_thread.start()