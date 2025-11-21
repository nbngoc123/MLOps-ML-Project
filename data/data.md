Hướng dẫn Dữ liệu Mẫu & Cấu trúc MinIO

Để các DAG Airflow chạy thành công, bạn cần lưu file CSV vào bucket nexusml theo cấu trúc thư mục dưới đây.

1. Phân loại Email (Email Classification)

Đường dẫn MinIO: nexusml/email_classifier/raw/emails.csv

Nội dung CSV:

subject,text,label
"Trúng thưởng lớn","Chúc mừng bạn đã trúng iPhone 15, click vào đây để nhận","spam"
"Hỗ trợ đăng nhập","Tôi quên mật khẩu tài khoản của mình, hãy giúp tôi","support"
"Đơn hàng #999","Đơn hàng của bạn đang được giao","order"
"Giảm giá thuốc","Mua thuốc giá rẻ nhất thị trường tại đây","spam"
"Lỗi thanh toán","Thẻ tín dụng của tôi bị từ chối khi checkout","support"
"Hóa đơn điện tử","Đây là hóa đơn cho đơn hàng #1002","order"


2. Phân loại Chủ đề (Topic Classification)

Đường dẫn MinIO: nexusml/topic_classifier/raw/news.csv

Nội dung CSV:

text,label
"Thị trường chứng khoán giảm điểm mạnh trong phiên hôm nay","business"
"Apple ra mắt dòng chip M3 mới cực mạnh","tech"
"Đội tuyển Việt Nam thắng trận vòng loại World Cup","sports"
"Lạm phát tại Mỹ đang có dấu hiệu hạ nhiệt","business"
"Trí tuệ nhân tạo đang thay đổi ngành lập trình","tech"
"Manchester United bổ nhiệm huấn luyện viên mới","sports"


3. Dịch / NLP Tiếng Việt (Vietnamese NLP)

Đường dẫn MinIO: nexusml/vietnamese_nlp/raw/vn_data.csv

Nội dung CSV:

text,target_lang
"Hôm nay trời đẹp quá, chúng ta đi chơi đi","en"
"Sản phẩm này giá bao nhiêu tiền vậy shop","en"
"Cảm ơn bạn rất nhiều","fr"
"Tôi không hài lòng với dịch vụ này","sentiment_negative"
"Món ăn này rất ngon","sentiment_positive"
"Xin chào tạm biệt và hẹn gặp lại","jp"


4. Gợi ý sản phẩm (Product Recommendation)

Đường dẫn MinIO: nexusml/product_recsys/raw/interactions.csv

Nội dung CSV:

user_id,product_id,rating
101,2001,5
101,2002,3
101,2005,4
102,2001,5
102,2003,2
103,2002,5
103,2004,4
104,2001,4
104,2005,5


Cách Upload thủ công (Nếu không dùng script)

Truy cập MinIO Console (thường là http://localhost:9001).

Tạo bucket tên là nexusml.

Tạo các folder con theo đúng tên: email_classifier, topic_classifier, vietnamese_nlp, product_recsys.

Trong mỗi folder đó, tạo folder con tên là raw.

Upload file CSV tương ứng vào folder raw.