import numpy as np
import openai
import base64
import numpy as np
import os

class Clustering():

    def __init__(self):
        super(Clustering, self).__init__()
        print('Clustering')
        self.api_key = ''                # REPLACE WITH YOUR API KEY
        openai.api_key = self.api_key



        self.prompt = """
            You are a runtime failure monitor for a vision-based autonomous robot navigating in an indoor environment.  
            Your task is to analyze a sequence of recent image observations, ending at the robot’s current position, and determine whether the robot is:

            - SAFE — confidently continuing in free space on a collision-free path, or  
            - UNSAFE — no free space ahead and at significant and credible risk of collision, based on observable evidence.

            Operational Context:

            - The robot must reach a predefined goal without any physical collisions.
            - You are provided with a time-ordered sequence of image frames, each after 0.5 seconds, ending at the robot’s current location.
            - The robot is moving with a maximum speed of 0.6 m/s.

            Evaluation Procedure:

            1. Predict Short-Term Trajectory  
               - Based on the image sequence, estimate the robot’s likely immediate direction of movement (e.g., straight, turning, drifting). Incorporate temporal cues for better motion understanding.

            2. Identify Relevant Obstacles  
               - Inspect the final image for physical objects that may intersect the predicted path. Focus only on nearby, collision-range elements that could plausibly interfere with the robot’s trajectory.

            3. Determine Collision Risk  
               - Mark the situation as unsafe if there is a visual alignment between the projected path and an obstacle, else mark it SAFE.

            4. Classify the Risk  
               - You are given the most common failure modes of this robot in the list below. If the risk matches one of the *Known Semantic Failure Reasons* listed below, return name of that exact label.  
               - If a new type of visible risk is present, briefly describe it in concise terms.  
               - If no substantial risk is visible along the projected path, mark it as SAFE.

            Known Semantic Failure Reasons:

            1. Name: Thin-Protruding Objects
               Description: Robot fails to detect or underestimates thin, low-contrast legs and bases, colliding with folding or office chairs, table/desk legs, etc.  
               Keywords: folding chair, foldable chair, thin metal legs, chair frame, chair legs, chair base, chair seat, office chair, casters, wheels, central post, desk leg, table leg, desk support

            2. Name: Uniform/Featureless Surfaces
               Description: Robot treats large flat, light-colored walls or cabinets as free space due to poor depth cues and lack of texture or edges.  
               Keywords: white cabinet, filing cabinet, locker/lockers, cabinet, uniform surface, featureless wall, light-colored surface, wall base, panel

            3. Name: Narrow-Gap/Clearance Misjudgment
               Description: Robot squeezes through tight passages—between obstacles or underestimates turning radius—leading to collisions in narrow spaces.  
               Keywords: narrow space, tight passage, narrow passage, misjudged gap, insufficient clearance, turning radius, misjudged space

            4. Name: Low-Height Clutter & Small Floor Obstacles
               Description: Robot runs into low-profile items on the floor (backpacks, cables, small debris) that blend into the background.  
               Keywords: backpack, cables, wires, power brick, small debris, equipment debris, low-lying object, floor clutter, soft object

            5. Name: Box-Like Equipment & Carts
               Description: Robot fails to detect or underestimates clearance around bulky, rectangular objects such as computer towers, servers, carts or pedestal-type furniture.  
               Keywords: computer tower, server, server cabinet, grey box, box-like object, equipment, pedestal, cart, machinery

            6. Name: Structural Edges: Door Frames, Jambs & Wall Corners
               Description: Robot collides with rigid vertical edges like door frames, jambs, wooden panels or cuts corners too sharply on wall intersections.  
               Keywords: door frame, door jamb, wooden frame/panel, frame edge, threshold, open door edge, corner, wall corner, external corner, protruding corner

            7. Name: Bins & Waste Receptacles
               Description: Robot collides with trash bins, recycling bins or their lids in tight spaces.  
               Keywords: trash bin, recycling bin, bin lid, blue-lidded bin, green bin, waste receptacle

            8. Name: Transparent & Reflective Surfaces
               Description: Robot mistakes glass doors, panels or mirrors for free space or is deceived by reflections.  
               Keywords: glass door, glass panel, mirror, reflective surface, transparent panel, reflection, deceptive surface

            9. Name: Overhead & Ceiling Fixtures
               Description: Robot drives into low-hanging fixtures or ceiling obstructions due to upward blind spots in its sensors.  
               Keywords: ceiling, low ceiling, fixture, overhead, overhead obstacle, piping, ceiling fixture

            Return only one of the following:
            - Name of a known semantic failure reason (exactly as written above)  
            - A brief description of a new failure type
            - The word SAFE

            Rules:

            - Do not provide explanations, justifications, or degrees of certainty.
            - Output must be a **single, definitive label**: one listed reason, a new concise reason, or SAFE.

            """




    def encode_image_to_base64(self, image_path):
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def cluster(self, images_path, idx):
        content = []
        for i in range(max(0, idx-1), idx+1):
            path = str(i) + '.png'
            print(path)
            base64_image = self.encode_image_to_base64(images_path + path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
                }
            })

        content.append({
            "type": "text",
            "text": self.prompt
        })
        messages = [{"role": "user", "content": content}]

        response = openai.ChatCompletion.create(
            # model="gpt-4o-mini",
            # model = "o1-mini",
            model = "o4-mini",
            messages=messages,
        )

        output = response['choices'][0]['message']['content']

        return output
