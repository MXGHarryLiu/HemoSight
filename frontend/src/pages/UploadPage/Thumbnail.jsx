import React from 'react';
import { Figure } from 'react-bootstrap';

class Thumbnail extends React.Component {
  // props: image (object), parent (Explorer)
  // fields of image (see also Explorer): 
  // image_id (string), filename (string), selected (bool), 
  // current (bool), label (list of strings), prob (list of floats),
  // status (string), reference_label (string)
  constructor(props) {
    super(props);
  }

  shouldComponentUpdate(nextProps, nextState) {
    // update if selected or caption changes
    return (
      this.props.image !== nextProps.image
    );
  }

  #renderCaption = () => {
    const { image } = this.props;
    if (image.status === 'error') {
      return `<p class="text-danger">Error</p>`;
    } else if (image.status === 'done') {
      return `<b>${image.label[0]}</b> (${image.reference_label})`;
    } else if (image.status === 'processing') {
      return `<p class="text-warning">Processing</p>`;
    } else if (image.status === 'pending') {
      return `<p class="text-info">Pending</p>`;
    } else {
      return 'Unknown';
    }
  };

  render() {
    const { image, parent } = this.props;
    const { image_id, selected, current, filename } = image;

    const caption = this.#renderCaption();

    const figureStyle = {
      backgroundColor: selected ? '#ADD8E6' : 'transparent',
      border: current ? '2px solid red' : '2px solid transparent' // Add border when current is true
    };
    
    const title = `${filename} (${image_id})`;

    return (
      <Figure onClick={() => parent.toggleSelected(image_id)} style={figureStyle}>
        <Figure.Image rounded
          src={`/api/query_image/${image_id}`}
          title={title}
          alt={title} />
        <Figure.Caption dangerouslySetInnerHTML={{ __html: caption }} />
      </Figure>
    );
  }
}

export default Thumbnail;
