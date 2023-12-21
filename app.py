import os, sys
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers import DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
import torch
import pika
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

load_dotenv(find_dotenv())

imgkit_priv_key = os.environ.get('IMGKIT_PRIV_KEY')
imgkit_pub_key = os.environ.get('IMGKIT_PUB_KEY')
imgkit_url_endpoint = os.environ.get('IMGKIT_URL_ENDPOINT')
mq_hostname = os.environ.get('MQ_HOSTNAME')
mq_port = os.environ.get('MQ_PORT')
mq_virt_host = os.environ.get('MQ_VIRT_HOST')
mq_username = os.environ.get('MQ_USERNAME')
mq_password = os.environ.get('MQ_PASSWORD')


def initialize_SDXL():
    global pipe, compel
    pipe = DiffusionPipeline.from_pretrained("stablediffusionapi/juggernaut-xl-v7", torch_dtype=torch.float16)
    pipe.to("cuda")

    # scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_pretrained(
    #     'stablediffusionapi/juggernaut-xl-v7',
    #     subfolder='scheduler',
    #     algorithm_type='sde-dpmsolver++',
    #     solver_order=2,
    #     # solver_type='heun' may give a sharper image. Cheng Lu reckons midpoint is better.
    #     solver_type='midpoint',
    #     use_karras_sigmas=True,
    #     )
    # # pipeline args documented here:
    # # https://github.com/huggingface/diffusers/blob/95b7de88fd0dffef2533f1cbaf9ffd9d3c6d04c8/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L548
    # temppipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
    #     'stablediffusionapi/juggernaut-xl-v7',
    #     scheduler=scheduler,
    #     torch_dtype=torch.float16,
    #     )
    # temppipe.to('cuda')
    # pipe = temppipe

    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    # TODO - Enable when Windows supports this
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

def initialize_ImageKit():
    global imagekit
    imagekit = ImageKit(
        private_key=imgkit_priv_key,
        public_key=imgkit_pub_key,
        url_endpoint=imgkit_url_endpoint
    )

def main():
    initialize_SDXL()
    initialize_ImageKit()

    credentials = pika.PlainCredentials(username=mq_username,password=mq_password,erase_on_connect=True)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=mq_hostname, port=mq_port, virtual_host=mq_virt_host, credentials=credentials))
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)
    channel.queue_declare(queue='prompts', arguments={'x-max-priority': 10})

    def callback(ch, method, properties, body):
        prompt = body.decode()
        print(f" [o] Received prompt '{prompt}'")
        additional_prompt = ", detailed background, highly intricate, elegant, professional fine detail, dramatic light, dynamic composition, focus, cute, great colors, perfect, ambient, vibrant, focused, sophisticated, creative, color spread, best, atmosphere, sharp, quality, very inspirational, colorful, epic, stunning, gorgeous, cinematic, fancy"
        generator = None
        if 'seed' in properties.headers:
            generator = torch.Generator(device="cuda").manual_seed(int(properties.headers['seed']))
        conditioning, pooled = compel(prompt + additional_prompt)
        output = pipe(prompt_embeds=conditioning,
                      pooled_prompt_embeds=pooled,
                      num_inference_steps=int(properties.headers.get('num_inference_steps', 20)), 
                      negative_prompt=properties.headers.get('negative_prompt',""), 
                      generator=generator,
                      guidance_scale=4)
        image = output.images[0]
        image.save(f"current.png")
        try:
            result = imagekit.upload_file(file=open("current.png", "rb"), # required
                                file_name=f"picture.png", # required
                                options=UploadFileRequestOptions(
                                    use_unique_file_name=True,
                                    tags=[datetime.utcnow().date().strftime('%Y-%m-%d'), datetime.utcnow().strftime('%H:%M')],
                                    folder=f"/endless-{properties.headers['category']}/",
                                    is_private_file=False,
                                    response_fields=['tags', 'is_private_file',
                                                    'embedded_metadata', 'custom_metadata'],
                                    webhook_url='https://webhook.site/ebce085f-5160-4bd8-af35-5fba222637a2',
                                    overwrite_tags=False,
                                    custom_metadata={'prompt': prompt},
                                )
                            )
            if result is None:
                print(f" [x] Failed to upload for unknown reason")
            else:
                print(f" [✓] Uploaded picture to '{result.url}'")
        except Exception as error:
            print(f" [x] Failed to upload: {error}")
        finally:
            ch.basic_ack(delivery_tag = method.delivery_tag)


    channel.basic_consume(queue='prompts',
                        auto_ack=False,
                        on_message_callback=callback)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


# prompt = "a cartoon kawaii cat sitting on a couch"

# # Generate an image from text
# image = pipe(prompt=prompt).images[0]

# # Save the image
# image.save("cat.png")


if __name__ == '__main__':
    while(1):
        try:
            main()
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as error:
            print(f" [x] Encountered Exception: {error}")
            print(" [⟳] Restarting consumer...")