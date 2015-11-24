A=imread('your_filename.jpg');
[u d v]=svd(double(A(:,:,3)));
k=3;
image( u(:,1:k)*d(1:k,1:k)*v(:,1:k)' );
c=(0:255)/255; colormap([c' c' c']); 
axis square;