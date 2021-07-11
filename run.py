from ocr.structured_data import detect_structured_data
from ocr.unstructured_data import detect_unstructured_data
import click
import cv2


@click.command()
@click.option("--image", prompt="Image file name", help="image file to run through ocr")
@click.option(
    "--structured", default=False, help="if the document is structured or unstructured"
)
def cli(image, structured):
    if structured:
        img = detect_structured_data(image)
    else:
        img = detect_unstructured_data(image)
    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cli()
