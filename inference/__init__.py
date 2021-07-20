import logging
import azure.functions as func
from inference.services.model_service import model


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Model Loading...')
    m = model()
    logging.info('Model Loading done !!')

    logging.info('Model inference...')
    if req.method != 'POST': return func.HttpResponse("Bad request", status_code=400)
    
    body_dict = req.files.to_dict()
    image_base64 = (body_dict['image'].stream.read())

    log, result = m.inference(image_base64)

    return func.HttpResponse(result, status_code=200)
