## NCAR won BEST TRAVEL AND TRANSPORTATION HACK at MakeUoft
Devpost: https://devpost.com/software/ncar-nighttime-collision-avoidance-system

## Inspiration
As OEMs(Original equipment manufacturers) and consumers keep putting on brighter and brighter lights, this can be blinding for oncoming traffic. Along with the fatigue and difficulty judging distance, it becomes increasingly harder to drive safely at night. Having an extra pair of night vision would be essential to protect your eyes and that's where the NCAR comes into play. The Nighttime Collision Avoidance Response system provides those extra sets of eyes via an infrared camera that uses machine learning to classify obstacles in the road that are detected and projects light to indicate obstacles in the road and allows safe driving regardless of the time of day.

## What it does
NCAR provides users with an affordable wearable tech that ensures driver safety at night
With its machine learning model, it can detect when humans are on the road when it is pitch black
The NCAR alerts users of obstacles on the road by projecting a beam of light onto the windshield using the OLED Display
If the user’s headlights fail, the infrared camera can act as a powerful backup light
## How we built it
- Machine Learning Model: Tensorflow API
- Python Libraries: OpenCV, PyGame
- Hardware: (Raspberry Pi 4B), 1 inch OLED display, Infrared Camera 
## Challenges we ran into
- Training machine learning model with limited training data
- Infrared camera breaking down, we had to use old footage of the ml model

## Accomplishments that we're proud of
- Implementing a model that can detect human obstacles from 5-7 meters from the camera
- building a portable design that can be implemented on any car

## What we learned
- Learned how to code different hardware sensors together
- Building a Tensorflow model on a Raspberry PI
- Collaborating with people with different backgrounds, skills and experiences

## What's next for NCAR: Nighttime Collision Avoidance System
- Building a more custom training model that can detect and interpolate the distances of the obstacles to the user
- A more sophisticated system of alerting users of obstacles on the path that is easy to maneuver
- Be able to adjust the OLED screen with a 3d printer to display light in a more noticeable way

