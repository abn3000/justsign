JustSign!


Demo URL: https://abn3000.github.io/justsign/

Note: Training has already been done ahead of time, so you can start playing immediately!

Inspiration  

We wanted to surprise our friend who recently lost his hearing by speaking to him in ASL. To make the learning process more engaging, we wanted to make a rhythm game that uses ASL signs instead of traditional button presses. This makes it easy to hop in and quickly practice your signs.

What it does  

JustSign! uses a list of constant letters that players have to sign at the right time to gain points. Signing early or late results in penalties, and the game gives feedback on how to improve. This gamification of learning makes it easier to hop on to the website and practice a couple signs.

How we built it  

JustSign! relies on a combination of open-source online models and our own manual training. We used Google's MediaPipe Hand Landmarking to track the hands; those are the lines you see when it's in use. We used Google's AI studio to get our API keys, which were then used to call Gemini and analyze the information received from MediaPipe. To actually design our site, we used Google's Antigravity and its related IDE to test the process of the app itself, and make the very first database for manual training. Without these tools, we would have had to detect the hands by ourselves and do much more of the manual coding.

Challenges we ran into  

The biggest challenge for us was getting the app to distinguish similar signs, such as T and M. Both of these involve poking a thumb through the fist, which can result in difficulty in identification. We solved this by manually making a database with a person doing all of the signs. We used over 20 photos for each letter with burst capture, resulting in a very high accuracy when combined with Gemini's analysis.

Accomplishments that we're proud of  

We're proud of making a product that analyzes the signs so quickly and accurately. As seen in our demo video, the program can keep up even with complex and quick signs. We all have some experience with HTML development, but we've never done anything quite as complex as this with the API keys.

What we learned  

We learned a lot about hand tracking by messing around with both mediapipe and researching the ASL database. We learned about coding HTML and integrating API keys to integrate agents into our work aswell.

What's next for JustSign!  

We'd like to improve on how well letters that require motion to sign (J, Z) are detected. After this, we want to expand JustSign! into full words, potentially using both hands to communicate.
