def generate_image(prompt):
    inputs = processor(prompt, return_tensors="pt")
    outputs = model(**inputs)
    image = outputs.images
    return image

prompt = "A turtle following a metal anchor to the bottom of the sea, fantasy, painting by Greg Rutkowski and Alphonse Mucha."
image = generate_image(prompt)